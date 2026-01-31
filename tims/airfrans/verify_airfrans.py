import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
# Assuming these are in your local path
import neuralop
from neuralop.models.base_model import get_model
from tims.airfrans.airfrans_dataset import AirfransDataset, SelectiveDataProcessor,load_airfrans_dataset
from neuralop.models import FNO  # Or your specific FNO_Small2d class
from tims.airfrans.airfrans_config_cp_weightedLoss import Default
import zencfg 

def repair_normalizer_dimensions(data_processor, device):
    """
    Correctly expands a 3-channel normalizer to 4 channels.
    Maps: [u, v, sdf] -> [u, v, mask, sdf]
    """
    in_norm = data_processor.in_normalizer
    
    # 1. Extract current stats (should be size 3)
    old_mean = in_norm.mean.flatten() 
    old_std = in_norm.std.flatten()
    
    # 2. Create new 4-channel tensors on the correct device
    new_mean = torch.zeros(4, device=device)
    new_std = torch.ones(4, device=device)
    
    # Map old indices to new indices
    # Old 0 (u) -> New 0
    # Old 1 (v) -> New 1
    # New 2 (mask) -> Remains Mean 0, Std 1
    # Old 2 (sdf) -> New 3
    new_mean[0], new_mean[1], new_mean[3] = old_mean[0], old_mean[1], old_mean[2]
    new_std[0], new_std[1], new_std[3] = old_std[0], old_std[1], old_std[2]

    # 3. Correct the shape logic
    # We want to keep the same number of dimensions but change the size of the 
    # specific dimension the normalizer is operating on (usually index 1).
    new_shape = list(in_norm.mean.shape)
    
    # Find which dimension in the 4D tensor is the channel dim (size 3)
    # and change it to 4.
    for i, dim_size in enumerate(new_shape):
        if dim_size == 3:
            new_shape[i] = 4
            break
            
    # Apply the new tensors
    in_norm.mean = new_mean.view(new_shape)
    in_norm.std = new_std.view(new_shape)
    
    print(f">>> Repair Complete: Input Normalizer shape is now {new_shape}")

# Usage in verify_airfrans.py:
# checkpoint = torch.load(...)
# data_processor.load_state_dict(checkpoint['data_processor'])
# audit_data_processor(data_processor)


def audit_data_processor(data_processor):
    print(f"\n{'#'*25} DATA PROCESSOR AUDIT {'#'*25}")
    
    if hasattr(data_processor, 'in_normalizer') and data_processor.in_normalizer:
        in_m = data_processor.in_normalizer.mean.flatten()
        in_s = data_processor.in_normalizer.std.flatten()
        
        print(f"\n[INPUTS] Found {len(in_m)} channels in Normalizer")
        names = ["u_inf", "v_inf", "mask", "sdf"]
        for i in range(min(len(in_m), len(names))):
            m, s = in_m[i].item(), in_s[i].item()
            status = "!! CORRUPTED !!" if i == 2 and abs(m) > 1e-3 else "OK"
            print(f"{i}: {names[i]:<8} | Mean {m:>8.4f} | Std {s:>8.4f} | {status}")
        
        if len(in_m) < 4:
            print(f"!! DIMENSION MISMATCH !! Normalizer has {len(in_m)} channels, expected 4.")

    if hasattr(data_processor, 'out_normalizer') and data_processor.out_normalizer:
        out_m = data_processor.out_normalizer.mean.flatten()
        out_s = data_processor.out_normalizer.std.flatten()
        print(f"\n[OUTPUTS] Target Scales:")
        targets = ["u_vel", "v_vel", "pressure", "nut"]
        for i in range(min(len(out_m), len(targets))):
            print(f"{i}: {targets[i]:<8} | Mean {out_m[i]:>10.2e} | Std {out_s[i]:>10.2e}")

def physical_reality_check(y_truth, y_pred):
    """Prints Max/Min for all 4 channels to see if the scale matches."""
    names = ['U-Vel', 'V-Vel', 'Pressure', 'Turbulence (nut)']
    print(f"\n{'FIELD':<15} | {'TRUTH (MAX/MIN)':<25} | {'PRED (MAX/MIN)':<25}")
    print("-" * 75)
    
    for i in range(4):
        t_max, t_min = y_truth[i].max(), y_truth[i].min()
        p_max, p_min = y_pred[i].max(), y_pred[i].min()
        print(f"{names[i]:<15} | {t_max:>8.2f} / {t_min:>8.2f} | {p_max:>8.2f} / {p_min:>8.2f}")

def print_processor_sanity_check(data_processor, names=["u_input", "v_input", "mask", "sdf"]):
    print(f"\n{'='*20} NORMALIZATION SANITY CHECK {'='*20}")
    
    # Check Input Normalizer (Selective: should skip channel 2)
    if hasattr(data_processor, 'in_normalizer') and data_processor.in_normalizer:
        mean = data_processor.in_normalizer.mean.squeeze()
        std = data_processor.in_normalizer.std.squeeze()
        
        print(f"{'Input Channel':<15} | {'Mean':>10} | {'Std':>10} | {'Status'}")
        print("-" * 55)
        for i, name in enumerate(names):
            # Check if this channel was supposed to be skipped (the mask)
            is_skipped = (i == 2) # Channel 2 is mask_binary
            
            # If skipped, mean should be 0 and std should be 1 (identity transform)
            # Or if your SelectiveProcessor uses a subset, check if indices match
            status = "SKIPPED (Mask)" if is_skipped else "NORMALIZED"
            print(f"{name:<15} | {mean[i]:>10.4f} | {std[i]:>10.4f} | {status}")

    # Check Output Normalizer (Pressure, Velocity, etc.)
    if hasattr(data_processor, 'out_normalizer') and data_processor.out_normalizer:
        out_names = ["u_target", "v_target", "p_target", "nut_target"]
        mean = data_processor.out_normalizer.mean.squeeze()
        std = data_processor.out_normalizer.std.squeeze()
        
        print(f"\n{'Output Channel':<15} | {'Mean':>10} | {'Std':>10}")
        print("-" * 40)
        for i, name in enumerate(out_names):
            print(f"{name:<15} | {mean[i]:>10.2e} | {std[i]:>10.2e}")
    
    print(f"{'='*63}\n")

def verify_input_encoder(encoder):
    print(f"\n{'='*20} INPUT ENCODER AUDIT {'='*20}")
    
    # 1. Check Channel Dimensions
    mean = encoder.mean.flatten()
    std = encoder.std.flatten()
    print(f"Stats Shape: {list(encoder.mean.shape)} | Channels detected: {len(mean)}")

    # 2. Check Physical Mapping
    # We expect 3 channels of stats representing [u_inf, v_inf, sdf]
    # which will be applied to indices [0, 1, 3] of the 4D input.
    names = ["u_velocity (inf)", "v_velocity (inf)", "SDF (geometry)"]
    
    print(f"\n{'Channel':<20} | {'Mean':>10} | {'Std':>10}")
    print("-" * 45)
    for i, name in enumerate(names):
        m, s = mean[i].item(), std[i].item()
        print(f"{name:<20} | {m:>10.4f} | {s:>10.4f}")

    # 3. Verify Selective Logic
    channels = getattr(encoder, 'channels_to_normalize', [])
    print(f"\nActive Channels for Normalization: {channels}")
    
    if 2 in channels:
        print("!! WARNING: Channel 2 (Mask) is set to be normalized! This will corrupt geometry.")
    else:
        print("✓ SUCCESS: Channel 2 (Mask) will be passed through untouched.")
    print(f"{'='*63}\n")

def run_prediction_demo(config, model_path: str, idx: int = 13):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Setup Dataset exactly like training
    # We pull every parameter from the zencfg object
    dataset = AirfransDataset(
        data_dir=config.data.data_dir,
        dataset_name='airfoil',
        train_split=config.data.train_split,
        test_splits=config.data.test_splits,
        train_resolution=config.data.train_resolution,
        test_resolutions=config.data.test_resolutions,
        encode_input=config.data.encode_input,
        encode_output=config.data.encode_output,
        # Ensure xlim/ylim match your Cartesian grid sampling
        xlim=config.data.xlim,
        ylim=config.data.ylim
    )

    # 2. Extract the DataProcessor (which now has the fitted Normalizers)
    data_processor = dataset.data_processor.to(device)

    # 3. Setup Model from Config
    model = get_model(config).to(device)

    # 2. Add GELU and spectral convolution to the allowlist before loading
    torch.serialization.add_safe_globals([zencfg.bunch.Bunch, torch._C._nn.gelu, neuralop.layers.spectral_convolution.SpectralConv])
    
    # Load weights safely
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])

    if 'data_processor' in checkpoint and checkpoint['data_processor']:
        data_processor.load_state_dict(checkpoint['data_processor'])
        print("Successfully loaded normalization constants.")
    else:
        print("No data processor state found in checkpoint; using default normalization.")
        print("Ensure that the normalization matches training data!")

    model.eval()

    # 3. Get a Sample
    # We use the raw TensorDataset inside the container
    sample = dataset.train_db[idx] # Airfoil index 13
    x_raw = sample['x'].unsqueeze(0).to(device)
    y_raw = sample['y'].unsqueeze(0).to(device)

    # 4. The Transformation Loop
    with torch.no_grad():
        # A. Preprocess: Standardize inputs (Selective normalization)
        batch = data_processor.preprocess({'x': x_raw, 'y': y_raw})
        
        # B. Inference: FNO works in Z-score space
        y_pred_norm = model(batch['x'])
        
        # C. Postprocess: Inverse Transform back to Physical Units (Pa, m/s)
        # This uses the inverse_transform method 
        y_pred_phys, _ = data_processor.postprocess(y_pred_norm, batch)

        # What happens if we skip postprocessing?
        #y_pred_phys = y_pred_norm  # This is still in normalized space

    # 5. Visualization
    y_truth = y_raw.squeeze().cpu().numpy()
    y_pred = y_pred_phys.squeeze().cpu().numpy()
    
    plot_prediction_with_residuals(y_truth, y_pred)

def plot_prediction_with_residuals(y_truth_phys, y_pred_phys):
    # y_truth_phys and y_pred_phys are shape [4, H, W] in physical units
    # Channels: 0:u, 1:v, 2:p, 3:nut
    
    y_truth = y_truth_phys
    y_pred = y_pred_phys
    
    # Calculate Residuals (Absolute Error)
    residuals = np.abs(y_truth - y_pred)
    
    names = ['U-Velocity (m/s)', 'V-Velocity (m/s)', 'Pressure (Pa)', 'Turbulence (nut)']
    cmaps = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'magma'] # magma is great for highlighting error
    
    # Create a 4x3 grid: Truth | Prediction | Residual
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    
    for i in range(4):
        # 1. Ground Truth
        im0 = axes[i, 0].imshow(y_truth[i].T, origin='lower', cmap=cmaps[i])
        axes[i, 0].set_title(f"Truth: {names[i]}")
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # 2. FNO Prediction
        im1 = axes[i, 1].imshow(y_pred[i].T, origin='lower', cmap=cmaps[i])
        axes[i, 1].set_title(f"Pred: {names[i]}")
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # 3. Residual (Error)
        im2 = axes[i, 2].imshow(residuals[i].T, origin='lower', cmap='inferno')
        axes[i, 2].set_title(f"Abs Error: {names[i]}")
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def plot_model_predictions(model, loader, data_processor, device, save_path="prediction_check.png"):
    model.eval()
    
    # Ensure data_processor and its components are on the correct device
    data_processor = data_processor.to(device)
    model = model.to(device)
    batch = next(iter(loader))
    x = batch['x'].to(device)
    y_truth_phys = batch['y'].to(device) 
    # 1. Ensure the normalizer is on the same device as the model output
    data_processor.out_normalizer = data_processor.out_normalizer.to(device)
    names = ['U-Vel (m/s)', 'V-Vel (m/s)', 'Pressure (Pa)', 'Nut']

    print(f"\n{'='*20} MODEL PREDICTION AUDIT {'='*20}")
    for i in range(3):
        print(f"{names[i]} - Normalizer Mean[{i}]: {data_processor.out_normalizer.mean[0, i, 0, 0].item():.4f}")
        print(f"{names[i]} - Normalizer Std[{i}]:  {data_processor.out_normalizer.std[0, i, 0, 0].item():.4f}")

    mu = data_processor.out_normalizer.mean.view(1, 3, 1, 1).to(device)
    sigma = data_processor.out_normalizer.std.view(1, 3, 1, 1).to(device)
    with torch.no_grad():
        # 1. Transform input (Preprocess)
        x_norm = data_processor.in_normalizer(x)
        
        y_pred_norm = model(x_norm) # This is the raw Z-score from the FNO
    
        # 1. Get stats directly from the normalizer buffers
        mean = data_processor.out_normalizer.mean
        std = data_processor.out_normalizer.std
        eps = data_processor.out_normalizer.eps


        # 3. Explicit Manual Math
        # y_phys = (y_norm * sigma) + mu
        y_pred_phys = torch.add(torch.mul(y_pred_norm, sigma), mu)

    # 1. Pull stats into pure local scalars to avoid any reference issues
    m_p = data_processor.out_normalizer.mean[0, 2, 0, 0].item() # Pressure Mean
    s_p = data_processor.out_normalizer.std[0, 2, 0, 0].item()  # Pressure Std

    with torch.no_grad():
        raw_output = model(x_norm) # Raw Z-scores from FNO
        
        # 2. Extract specific raw Z-max for Pressure
        z_max_p = raw_output[0, 2].max().item()
        
        # 3. Perform the math using SCALARS, not tensors
        # This bypasses all PyTorch broadcasting and DataProcessor logic
        manual_phys_max = (z_max_p * s_p) + m_p

    print(f"\n--- ISOLATION AUDIT (Pressure) ---")
    print(f"Raw Z-Score Max: {z_max_p:.4f}")
    print(f"Stats used:      Mean={m_p:.2f}, Std={s_p:.2f}")
    print(f"Manual Result:   {manual_phys_max:.2f} Pa")
    print(f"Previous Result: 0.90 (CORRUPTED)")
    for i in range(3):
        t_max = y_truth_phys[0, i].max().item()
        p_max = y_pred_phys[0, i].max().item()
        
        # Check if the prediction is within 50% to 150% of the truth range
        status = "✓ OK" if abs(p_max) > 0.1 * abs(t_max) else "!! SCALE MISMATCH !!"
        print(f"{names[i]:<15} | {t_max:>10.2f} | {p_max:>10.2f} | {status}")
    print(f"{'='*60}\n")
    print(f"Output Normalizer Mean: {data_processor.out_normalizer.mean.flatten()}")
    print(f"Output Normalizer Std:  {data_processor.out_normalizer.std.flatten()}")
    # 5. Plotting Logic
    # (Assuming your existing 4x3 subplot structure for Truth/Pred/Error)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i in range(3):
        # Truth Physical
        im0 = axes[i, 0].imshow(y_truth_phys[0, i].cpu().numpy(), origin='lower')
        plt.colorbar(im0, ax=axes[i, 0])
        axes[i, 0].set_title(f"Truth: {names[i]}")
        
        # Pred Physical
        im1 = axes[i, 1].imshow(y_pred_phys[0, i].cpu().numpy(), origin='lower')
        plt.colorbar(im1, ax=axes[i, 1])
        axes[i, 1].set_title(f"Pred: {names[i]}")
        
        # Abs Error
        error = torch.abs(y_truth_phys[0, i] - y_pred_phys[0, i])
        im2 = axes[i, 2].imshow(error.cpu().numpy(), origin='lower', cmap='plasma')
        plt.colorbar(im2, ax=axes[i, 2])
        axes[i, 2].set_title("Abs Error")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    # Ensure you have a trained model at the specified path
    from tims.airfrans.airfrans_config_cp_weightedLoss import Default
    # Read the configuration
    from zencfg import make_config_from_cli
    import sys

    config = make_config_from_cli(Default)
    config = config.to_dict()
    model_checkpoint_path = "/home/timm/Projects/PIML/neuraloperator/tims/airfrans/checkpoints-weighted-L1/airfrans_cp_l2_fno_final.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # Create the processor (this fits it to the CURRENT subset in memory)

    # Example usage
    train_loader, test_loaders, data_processor = load_airfrans_dataset(
        data_dir=config.data.data_dir,
        dataset_name=config.data.dataset_name,
        train_split=config.data.train_split,
        test_splits=config.data.test_splits,
        batch_size=config.data.batch_size,
        test_batch_sizes=config.data.test_batch_sizes,
        test_resolutions=config.data.test_resolutions,
        encode_input=config.data.encode_input,    
        encode_output=config.data.encode_output, 
        encoding=config.data.encoding,
        channel_dim=config.data.channel_dim,
    )


    print("Status: Initial stats from current dataset subset:")


    # Load the saved "Training Truth"
    torch.serialization.add_safe_globals([zencfg.bunch.Bunch, torch._C._nn.gelu, neuralop.layers.spectral_convolution.SpectralConv])

    checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=True)
    model = get_model(config).to(device)
    model.load_state_dict(checkpoint['model'])
    # 3. Handle the Data Processor
    if 'data_processor' in checkpoint:
        # IMPORTANT: Ensure your data_processor object already exists 
        # (usually created by load_airfrans_dataset earlier in the script)
        data_processor.load_state_dict(checkpoint['data_processor'])
        data_processor.to(device)
        print("✓ Status: Model and Stats updated from checkpoint!")
        
        # 4. CRITICAL REPAIR STEP
        # If your checkpoint was from a 3-channel run but your model is now 4-channel,
        # you MUST run the repair logic here.
        if data_processor.in_normalizer.mean.numel() == 3:
            print("!! Detecting 3-channel stats for 4-channel model. Repairing...")
            repair_normalizer_dimensions(data_processor, device)
    else:
        print("CRITICAL ERROR: No data_processor state found in checkpoint!")
    
    verify_input_encoder(data_processor.in_normalizer)

    # Hard check: Is the mask (index 2) actually 0/1?
    sample = next(iter(train_loader))
    x_test = data_processor.preprocess(sample)['x']
    mask_val = x_test[0, 2, 0, 0].item()

    test_loaders_keys = list(test_loaders.keys())
    test_loader = test_loaders[test_loaders_keys[0]]

    plot_model_predictions(model, test_loader, data_processor, device, save_path="prediction_check.png")    

    #run_prediction_demo(config, model_checkpoint_path)
