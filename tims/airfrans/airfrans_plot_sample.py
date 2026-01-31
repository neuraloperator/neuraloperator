import matplotlib.pyplot as plt
import torch
import numpy as np
import json
# Read the configuration
from zencfg import make_config_from_cli
import sys

import zencfg

sys.path.insert(0, "../")
import neuralop
import neuralop
from neuralop.models.base_model import get_model
from tims.airfrans.airfrans_config_cp_weightedLoss import Default
from pathlib import Path
from tims.airfrans.airfrans_dataset import load_airfrans_dataset

def plot_sample_full_pipeline(model, loader, data_processor, device, save_path="pipeline_audit.png"):
    model.eval()
    
    # 1. Get a single sample
    batch = next(iter(loader))
    x_raw = batch['x'][0:1].to(device)  # Raw physical inputs [1, 4, H, W]
    y_raw = batch['y'][0:1].to(device)  # Raw physical targets [1, 3, H, W] (u, v, Cp)
    
    with torch.no_grad():
        # 2. Step 1: Normalization (Pre-processing)
        x_norm = data_processor.preprocess({'x': x_raw, 'y': y_raw})['x']
        if isinstance(x_norm, torch.Tensor):
            x_norm = x_norm.to(device)
        elif isinstance(x_norm, dict):
            x_norm = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x_norm.items()}

        y_norm_pred = model(x_norm)
        
        # 3. Step 2: Inference (Z-score domain)
        y_norm_pred = model(x_norm)
        
        # 4. Step 3: Denormalization (Post-processing)
        data_processor = data_processor.to(device)
        y_phys_pred = data_processor.postprocess(y_norm_pred)

    out_labels = ['U-Velocity', 'V-Velocity', 'Pressure Coeff (Cp)']
    in_labels = ['U-inf', 'V-inf', 'Mask', 'SDF']

    # Creating a 4x5 grid to show Inputs and the full Output pipeline
    fig, axes = plt.subplots(4, 5, figsize=(25, 18))
    
# --- ROW 1: INPUT CHANNELS ---
    for i in range(4):
        im_in_raw = axes[i, 0].imshow(x_raw[0, i].cpu(), origin='lower')
        plt.colorbar(im_in_raw, ax=axes[i, 0])
        axes[i, 0].set_title(f"Raw Input: {in_labels[i]}")
        
        im_in_norm = axes[i, 1].imshow(x_norm[0, i].cpu(), origin='lower', cmap='RdBu_r')
        plt.colorbar(im_in_norm, ax=axes[i, 1])
        axes[i, 1].set_title(f"Norm Input: {in_labels[i]}")

    # --- ROW 2-4: OUTPUTS (Truth vs Norm Pred vs Phys Pred) ---
# --- ROWS 1-3: OUTPUT COMPARISON (u, v, Cp) ---
    for i in range(3):
        # Column 2: Ground Truth Physical
        im_truth = axes[i, 2].imshow(y_raw[0, i].cpu(), origin='lower')
        plt.colorbar(im_truth, ax=axes[i, 2])
        axes[i, 2].set_title(f"Truth Phys: {out_labels[i]}")
        
        # Column 3: Z-Score Prediction (What FNO actually outputs)
        im_norm_p = axes[i, 3].imshow(y_norm_pred[0, i].cpu(), origin='lower', cmap='plasma')
        plt.colorbar(im_norm_p, ax=axes[i, 3])
        axes[i, 3].set_title(f"Norm Pred (Z): {out_labels[i]}")
        
        # Column 4: Final Physical Prediction
        vmin, vmax = y_raw[0, i].min().cpu(), y_raw[0, i].max().cpu()
        im_phys_p = axes[i, 4].imshow(y_phys_pred[0, i].cpu(), origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(im_phys_p, ax=axes[i, 4])
        axes[i, 4].set_title(f"Phys Pred: {out_labels[i]}")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Full pipeline plot saved to {save_path}")
    plt.close()

def audit_dataset_pipeline(loader, data_processor, device, save_path="data_audit_full.png"):
    """
    Demonstrates the full data pipeline:
    Raw Input -> Norm Input | Raw Target -> Norm Target -> Reconstructed Target
    """
    # 1. Get a single sample
    batch = next(iter(loader))
    x_raw = batch['x'][0:1].to(device)  
    y_raw = batch['y'][0:1].to(device)  
    
    # 2. Preprocess (Normalization)
    processed_dict = data_processor.preprocess({'x': x_raw, 'y': y_raw})
    x_norm = processed_dict['x']
    y_norm_truth = processed_dict['y'] # These are the Z-scores the model tries to predict
    
    # 3. Postprocess (Denormalization)
    y_phys_reconstructed = data_processor.postprocess(y_norm_truth)

    in_labels = ['U-inf', 'V-inf', 'Mask', 'SDF']
    out_labels = ['U-Velocity', 'V-Velocity', 'Pressure Coeff (Cp)']

    # 5 columns now: Raw In | Norm In | Raw Out | Norm Out (Z) | Reconstructed Out
    fig, axes = plt.subplots(4, 5, figsize=(25, 16))
    plt.suptitle("Data Pipeline Audit: Full Input/Output Normalization Check", fontsize=18)

    # --- INPUT PIPELINE (Rows 0-3, Columns 0-1) ---
    for i in range(4):
        # Raw Input
        im0 = axes[i, 0].imshow(x_raw[0, i].cpu(), origin='lower',cmap='plasma')
        plt.colorbar(im0, ax=axes[i, 0])
        axes[i, 0].set_title(f"RAW IN: {in_labels[i]}")
        
        # Normalized Input
        im1 = axes[i, 1].imshow(x_norm[0, i].cpu(), origin='lower', cmap='plasma')
        plt.colorbar(im1, ax=axes[i, 1])
        axes[i, 1].set_title(f"NORM IN: {in_labels[i]}")

    # --- OUTPUT PIPELINE (Rows 0-2, Columns 2-4) ---
    for i in range(3):
        # Raw Target (Physics)
        im2 = axes[i, 2].imshow(y_raw[0, i].cpu(), origin='lower')
        plt.colorbar(im2, ax=axes[i, 2])
        axes[i, 2].set_title(f"RAW OUT: {out_labels[i]}")
        
        # Normalized Target (The Z-score the model actually sees)
        # Using plasma/inferno to distinguish Z-scores from physical units
        im3 = axes[i, 3].imshow(y_norm_truth[0, i].cpu(), origin='lower', cmap='plasma')
        plt.colorbar(im3, ax=axes[i, 3])
        axes[i, 3].set_title(f"NORM OUT (Z): {out_labels[i]}")
        
        # Reconstructed Target (Should be identical to Column 2)
        im4 = axes[i, 4].imshow(y_phys_reconstructed[0, i].cpu(), origin='lower')
        plt.colorbar(im4, ax=axes[i, 4])
        axes[i, 4].set_title(f"RECONSTRUCTED: {out_labels[i]}")
        
    # Clean up empty subplots
    for row in range(4):
        for col in range(2, 5):
            if row == 3: axes[row, col].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"Detailed audit plot saved to {save_path}")



def plot_normalized_comparison(model, loader, data_processor, device, save_path="normalized_target_comparison.png"):
    model.eval()
    data_processor = data_processor.to(device)
    
    # 1. Fetch sample
    batch = next(iter(loader))
    x_raw = batch['x'][0:1].to(device)
    y_raw = batch['y'][0:1].to(device)
    
    with torch.no_grad():
        # 2. Get Normalized Truth (Z-score)
        # We manually call preprocess to see what the model 'sees' as target
        processed = data_processor.preprocess({'x': x_raw, 'y': y_raw})
        x_norm = processed['x']
        y_norm_truth = processed['y'] # This is the target in Z-score domain
        
        # 3. Get Normalized Prediction
        y_norm_pred = model(x_norm)
        
        # 4. Get Physical Prediction
        y_phys_pred = data_processor.postprocess(y_norm_pred)

    out_labels = ['U-Velocity', 'V-Velocity', 'Cp']
    
    # Grid: 3 Rows (u, v, Cp) x 4 Columns
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    for i in range(3):
        # --- COLUMN 1: Normalized Truth (Z) ---
        im1 = axes[i, 0].imshow(y_norm_truth[0, i].cpu(), origin='lower', cmap='RdBu_r')
        plt.colorbar(im1, ax=axes[i, 0])
        axes[i, 0].set_title(f"Truth Norm (Z): {out_labels[i]}")
        
        # --- COLUMN 2: Normalized Pred (Z) ---
        # Using same color limits as Truth Norm for direct comparison
        z_min, z_max = y_norm_truth[0, i].min().cpu(), y_norm_truth[0, i].max().cpu()
        im2 = axes[i, 1].imshow(y_norm_pred[0, i].cpu(), origin='lower', cmap='RdBu_r', vmin=z_min, vmax=z_max)
        plt.colorbar(im2, ax=axes[i, 1])
        axes[i, 1].set_title(f"Pred Norm (Z): {out_labels[i]}")
        
        # --- COLUMN 3: Physical Truth ---
        im3 = axes[i, 2].imshow(y_raw[0, i].cpu(), origin='lower')
        plt.colorbar(im3, ax=axes[i, 2])
        axes[i, 2].set_title(f"Truth Phys: {out_labels[i]}")
        
        # --- COLUMN 4: Physical Prediction ---
        # Using physical limits from Truth Phys
        p_min, p_max = y_raw[0, i].min().cpu(), y_raw[0, i].max().cpu()
        im4 = axes[i, 3].imshow(y_phys_pred[0, i].cpu(), origin='lower', vmin=p_min, vmax=p_max)
        plt.colorbar(im4, ax=axes[i, 3])
        axes[i, 3].set_title(f"Pred Phys: {out_labels[i]}")

    plt.suptitle(f"Z-Score vs Physical Domain Audit", fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Audit plot saved to {save_path}")
    plt.close()

def print_processor_stats(data_processor):
    # Retrieve means and stds
    in_mean = data_processor.in_normalizer.mean.flatten()
    in_std = data_processor.in_normalizer.std.flatten()
    out_mean = data_processor.out_normalizer.mean.flatten()
    out_std = data_processor.out_normalizer.std.flatten()

    in_labels = ['u_inf', 'v_inf', 'mask', 'sdf']
    out_labels = ['u_velocity', 'v_velocity', 'Cp']

    print(f"\n{'='*25} INPUT NORMALIZER STATS {'='*25}")
    print(f"{'Channel':<15} | {'Mean':>12} | {'Std':>12} | {'Status'}")
    print("-" * 68)
    
    # Use the length of the actual mean tensor to avoid IndexErrors
    for i in range(len(in_mean)):
        label = in_labels[i] if i < len(in_labels) else f"Ch_{i}"
        m, s = in_mean[i].item(), in_std[i].item()
        
        # Identify if a mask is improperly normalized
        if i == 2 and (abs(m) > 1e-5 or abs(s - 1.0) > 1e-5):
            status = "!! CORRUPTED !!" 
        elif i == 2:
            status = "PASSTHROUGH"
        else:
            status = "NORMALIZING"
            
        print(f"{label:<15} | {m:>12.4f} | {s:>12.4f} | {status}")

    print(f"\n{'='*25} OUTPUT NORMALIZER STATS {'='*25}")
    # ... (same loop logic for output) ...

def verify_input_encoder(encoder):
    print(f"\n{'='*20} INPUT ENCODER AUDIT {'='*20}")
    
    mean = encoder.mean.flatten()
    std = encoder.std.flatten()
    
    # Correct mapping for 4-channel input [u, v, mask, sdf]
    all_names = ["u_inf", "v_inf", "Mask (Binary)", "SDF (Geometry)"]
    
    print(f"{'Index':<5} | {'Channel':<15} | {'Mean':>10} | {'Std':>10}")
    print("-" * 50)
    for i in range(len(mean)):
        m, s = mean[i].item(), std[i].item()
        name = all_names[i] if i < len(all_names) else f"Unknown_{i}"
        print(f"{i:<5} | {name:<15} | {m:>10.4f} | {s:>10.4f}")

    channels = getattr(encoder, 'channels_to_normalize', [])
    print(f"\nActive Channels for Normalization: {channels}")
    print(f"{'='*63}\n")

def verify_output_encoder(encoder):
    print(f"\n{'='*20} OUTPUT ENCODER AUDIT {'='*20}")
    
    mean = encoder.mean.flatten()
    std = encoder.std.flatten()
    
    # Correct mapping for 4-channel input [u, v, mask, sdf]
    all_names = ["u_target", "v_target", "Cp", "nut"]
    
    print(f"{'Index':<5} | {'Channel':<15} | {'Mean':>10} | {'Std':>10}")
    print("-" * 50)
    for i in range(len(mean)):
        m, s = mean[i].item(), std[i].item()
        name = all_names[i] if i < len(all_names) else f"Unknown_{i}"
        print(f"{i:<5} | {name:<15} | {m:>10.4f} | {s:>10.4f}")


    print(f"{'='*63}\n")

if __name__ == "__main__":

    # Make sure we only print information when needed
    config = make_config_from_cli(Default)
    config = config.to_dict()

    # Print configuration details
    if config.verbose:
        print(f"##### CONFIG #####\n")
        print(json.dumps(config, indent=4))


    # Load the Airfrans dataset
    data_dir = Path(config.data.data_dir).expanduser()

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
            channel_dim=1,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")




    model_checkpoint_path = "/home/timm/Projects/PIML/neuraloperator/tims/airfrans/checkpoints-weighted-L1/airfrans_cp_l2_fno_final.pt"
    
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
        
    else:
        print("CRITICAL ERROR: No data_processor state found in checkpoint!")


    print_processor_stats(data_processor)
    verify_input_encoder(data_processor.in_normalizer)
    verify_output_encoder(data_processor.out_normalizer)
    audit_dataset_pipeline(
        loader=test_loaders[config.data.test_resolutions[0]],
        data_processor=data_processor,
        device=device,
        save_path="data_pipeline_audit.png"
    )
    plot_sample_full_pipeline(model, test_loaders[config.data.test_resolutions[0]], data_processor, device, save_path="predited_pipeline_audit.png")

    plot_normalized_comparison(model=model,
        loader=test_loaders[config.data.test_resolutions[0]],
        data_processor=data_processor,
        device=device,
        save_path="normalized_target_comparison.png"
    )
