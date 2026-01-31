from logging import config
import torch
import neuralop
from neuralop import data
from neuralop.data.datasets.pt_dataset import PTDataset
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
import json
from neuralop.data.datasets.tensor_dataset import TensorDataset
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import zencfg
from neuralop.models.base_model import get_model


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


class SelectiveUnitGaussianNormalizer(UnitGaussianNormalizer):
    def __init__(self,  dim, eps=1e-5, channels_to_normalize=[0, 1, 3], mask_channel=2):
        # Call the parent constructor
        super().__init__(mean=None, std=None, eps=eps, dim=dim)
        
        # This is where it lives!
        self.channels_to_normalize = channels_to_normalize
        self.mask_channel = mask_channel

    def transform(self, x):
        # x is [Batch, 4, H, W]       
        # Ensure mean and std are specifically the same shape as the slice
        # If your fit logic produced [1, 3, 1, 1], this will work.
        # If it produced [1, 4, 1, 1], you must slice the stats too.
        
        m = self.mean
        s = self.std
        
        # 1. Full Tensor Case (Standard for Outputs)
        # If the model output is 3 channels and the normalizer has 3 channels
        if x.shape[1] == m.shape[1]:
            return (x - m) / (s + self.eps)
        
        # 2. Selective Case (Standard for Inputs)
        # If the input is 4 channels but we only want to normalize 3 of them
        else:
            channels = self.channels_to_normalize # [0, 1, 3]

            x_norm = x.clone()
            
            # Slicing the stats to match the selected channels
            m_sub = m[:, channels, ...]
            s_sub = s[:, channels, ...]
            
            x_norm[:, channels, ...] = (x[:, channels, ...] - m_sub) / (s_sub + self.eps)
            return x_norm

    def inverse_transform(self, x):
        m = self.mean
        s = self.std
        # Check if the normalizer stats match the full input or the subset
        # should match for outputs  not for inputs where mask is passed through
        if x.shape[1] == m.shape[1]:

            return (x*(s + self.eps)) + m
        else:
            channels = self.channels_to_normalize
            x_phys = x.clone()

            m = m[:, channels, ...]
            s = s[:, channels, ...]
            x_phys[:, channels, ...] = (x[:, channels, ...] * (s + self.eps)) + m
            return x_phys
    

class SelectiveDataProcessor(DefaultDataProcessor):
    def preprocess(self, data, batched=True):
        x = data['x'].to(self.device)
        y = data['y'].to(self.device)

        # keep the nut channel 
        y = y[:, :, :, :]
        
        # --- THE HARD GATE ---
        # 1. Save the raw binary mask (Channel 2)
        raw_mask = x[:, 2:3, :, :].clone()

        if self.in_normalizer is not None:
            # We pass the WHOLE x to the normalizer. 
            # Our new SelectiveNormalizer above will handle the [0, 1, 3] slicing internally.
            x = self.in_normalizer(x)
        
        # 2. Force the mask back to raw 0/1 values ( and flip it)
        x[:, 2:3, :, :] = 1 - raw_mask
        
        if self.out_normalizer is not None:
            y = self.out_normalizer(y)
            
        return {'x': x, 'y': y}

    def postprocess(self, out, data=None, batched=True):
        if self.out_normalizer is not None:
            out = self.out_normalizer.inverse_transform(out)
        
        # Ensure truth data is on same device as prediction for Loss/Metric calc
        if data is not None:
            if hasattr(out, 'device'):
                for key in data:
                    if hasattr(data[key], 'to'):
                        data[key] = data[key].to(out.device)
        
            return out, data
        return out

class AirfransDatasetAll(PTDataset):
    def __init__(
        self,
        data_dir: Union[Path, str],
        dataset_name: str = 'airfoil_cp',
        train_split: str = 'aoa_train',
        test_splits: List[str] = ['aoa_test','aoa_test'],
        batch_size: int = 32,
        test_batch_sizes: List[int] = [32, 32],
        train_resolution: int = 128,
        test_resolutions: List[int] = [128,256],
        xlim: float = 2.0,
        ylim: float = 2.0,
        encode_input: bool = True,
        encode_output: bool = True,
        encoding: str = "channel-wise",
        channel_dim: int = 1,
        channels_squeezed: bool = False,  # Our data already has explicit channel dims
    ):
        """
        Initialize AirfoilDataset with AirFRANS-specific parameters.
        
        Args:
            train_split: Split name for training ('full_train', 'scarce_train', 'aoa_train', 'reynolds_train')
            test_splits: List of split names for testing ( 'full_test', 'aoa_test', 'reynolds_test')
            xlim, ylim: Domain size parameters
            ... other standard PTDataset parameters
        """
        
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        
        # Load manifest to get actual sample counts
        manifest_path = data_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
        else:
            raise FileNotFoundError(f"Manifest file not found at {manifest_path}")
        
        # Calculate actual sample counts from manifest
        train_foil_name = manifest_data.get(train_split, [])
        n_train = len([d for d in train_foil_name ])
        
        n_tests = []
        for test_split in test_splits:
            test_foil_name = manifest_data.get(test_split, [])
            n_test = len([d for d in test_foil_name ])
            n_tests.append(n_test)
        
        # Store AirFRANS-specific parameters
        self.train_split = train_split
        self.test_splits = test_splits
        self.xlim = xlim
        self.ylim = ylim
        
        # Store dataloader properties (from PTDataset)
        self.batch_size = batch_size
        self.test_resolutions = test_resolutions
        self.test_batch_sizes = test_batch_sizes
        
        # Load training data with custom filename
        train_file = data_dir / f"{dataset_name}_{train_split}_{train_resolution}.pt"
        train_data = torch.load(train_file.as_posix())
        
        x_train = train_data["x"].type(torch.float32).clone()
        y_train = train_data["y"].clone()

        y_train = y_train[:, :, :, :]  # [B, 4, H, W] keep all 4 channels including nut

        print(f"Loading train db for {train_split} resolution {train_resolution} with {n_train} samples")

        print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

        print(f"Input channels : {x_train.shape[1]}")
        print(f"Output channels: {y_train.shape[1]}")

        del train_data
        
        # Fit optional encoders to train data (from PTDataset logic)
        # For inputs: normalize u_input, v_input, sdf_fixed but NOT mask_binary (channel 2)
        if encode_input:
            if encoding == "channel-wise":
                reduce_dims = list(range(x_train.ndim))
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]
            
            # Create separate normalizers for different channels
            # Channels: [u_input, v_input, mask_binary, sdf_fixed]
            #           [   0   ,    1   ,      2     ,    3    ]
            channels_to_normalize = [0, 1, 3]  # Skip channel 2 (mask_binary)
            input_encoder = SelectiveUnitGaussianNormalizer(dim=reduce_dims, eps=1e-5, channels_to_normalize=channels_to_normalize, mask_channel=2)
            # feed whole x_train, the normalizer will handle selective normalization
            input_encoder.fit(x_train)
            # force the mask channel stats to 0 mean, 1 std
            input_encoder.mean[:, 2, ...] = 0.0
            input_encoder.std[:, 2, ...] = 1.0

            
            # Store which channels to normalize
            input_encoder.channels_to_normalize = channels_to_normalize
            input_encoder.mask_channel = 2  # Store mask channel index

            print(f"✓ Input Normalizer fitted on channels {channels_to_normalize}")
            print(f"✓ Mask channel {input_encoder.mask_channel} will be passed through.")
            print(f"Input Normalizer mean shape: {input_encoder.mean.shape}, std shape: {input_encoder.std.shape}")
            print(f"Input encoder mean: {input_encoder.mean}")
            print(f"Input encoder std: {input_encoder.std}")
            print(f"Input encoder mask channel: {input_encoder.mask_channel}")
        else:
            input_encoder = None

        if encode_output:
            if encoding == "channel-wise":
                reduce_dims = list(range(y_train.ndim))
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]
            output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            output_encoder.fit(y_train)
        else:
            output_encoder = None
        
        # Create train dataset
        self._train_db = TensorDataset(x_train, y_train)
        
        # Create custom data processor that handles selective input normalization
        self._data_processor = SelectiveDataProcessor(
            in_normalizer=input_encoder, 
            out_normalizer=output_encoder
        )
        
        # Load test data with custom filenames
        self._test_dbs = {}
        for res, n_test, test_split in zip(test_resolutions, n_tests, test_splits):
            print(f"Loading test db for {test_split} resolution {res} with {n_test} samples")
            
            test_file = data_dir / f"{dataset_name}_{test_split}_{res}.pt"
            test_data = torch.load(test_file.as_posix())
            
            x_test = test_data["x"].type(torch.float32).clone()
            y_test = test_data["y"].clone()
            
            del test_data
            
            test_db = TensorDataset(x_test, y_test)
            self._test_dbs[res] = test_db
    # Properties are inherited from PTDataset, so you don't need to redefine them
    # unless you want to add custom behavior



def load_airfrans_dataset(
    data_dir,
    dataset_name,    
    train_split,
    test_splits,
    batch_size,
    test_batch_sizes,

    train_resolution=64,
    test_resolutions=[64, 128, 256],
    encode_input=True,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,
):
    dataset = AirfransDatasetAll(
        data_dir=data_dir,
        dataset_name=dataset_name,
        train_split=train_split,
        test_splits=test_splits,
        batch_size=batch_size,
        test_batch_sizes=test_batch_sizes,
        train_resolution=train_resolution,
        test_resolutions=test_resolutions,
        encode_input=encode_input,
        encode_output=encode_output,
        channel_dim=channel_dim,
        encoding=encoding,
    )

    # return dataloaders for backwards compat
    train_loader = DataLoader(
        dataset.train_db,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
    )

    test_loaders = {}
    for res, test_bsize in zip(test_resolutions, test_batch_sizes):
        print(f"Creating DataLoader for test resolution {res} with batch size {test_bsize}")
        test_loaders[res] = DataLoader(
            dataset.test_dbs[res],
            batch_size=test_bsize,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            persistent_workers=False,
        )

    return train_loader, test_loaders, dataset.data_processor


def load_trained_model(config, model_path, device='cuda'):
    # Instantiate the FNO model from your config
    # e.g., model = FNO_Small2d(data_channels=4, out_channels=4...)
    model = get_model(config).to(device)
    
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def plot_model_predictions(model, dataset, data_processor, sample_idx=0, device='cuda'):
    # 1. Get raw data
    raw_sample = dataset.train_db[sample_idx]
    x_raw = raw_sample['x'].unsqueeze(0).to(device) # (1, C, H, W)
    y_raw = raw_sample['y'].unsqueeze(0).to(device)
    
    # 2. Normalize inputs for the model
    # We only care about x_norm here
    processed = data_processor.preprocess({'x': x_raw, 'y': y_raw})
    x_norm = processed['x']
    
    # 3. Model Inference (Prediction is in Z-score space)
    with torch.no_grad():
        y_pred_norm = model(x_norm)
        
    # 4. Inverse Transform to Physical Units
    # We use our updated postprocess to get Pascals/Velocity back
    y_pred_phys, _ = data_processor.postprocess(y_pred_norm, processed)
    
    # Convert to numpy for matplotlib
    # Channels for y: 0:u, 1:v, 2:p, 3:nut
    y_ground_truth = y_raw.squeeze(0).cpu().numpy()
    y_prediction = y_pred_phys.squeeze(0).cpu().numpy()
    
    output_names = ['u_target (m/s)', 'v_target (m/s)', 'p_target (Pa)', 'nut_fixed']
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 15))
    
    for i in range(4):
        # Column 1: Ground Truth (Physical)
        im0 = axes[i, 0].imshow(y_ground_truth[i].T, origin='lower', cmap='RdBu_r')
        axes[i, 0].set_title(f"Truth: {output_names[i]}")
        fig.colorbar(im0, ax=axes[i, 0])
        
        # Column 2: Model Prediction (Physical)
        im1 = axes[i, 1].imshow(y_prediction[i].T, origin='lower', cmap='RdBu_r')
        axes[i, 1].set_title(f"FNO Prediction: {output_names[i]}")
        fig.colorbar(im1, ax=axes[i, 1])
        
        # Column 3: Absolute Error
        error = np.abs(y_ground_truth[i] - y_prediction[i])
        im2 = axes[i, 2].imshow(error.T, origin='lower', cmap='inferno')
        axes[i, 2].set_title(f"Abs Error: {output_names[i]}")
        fig.colorbar(im2, ax=axes[i, 2])

    plt.tight_layout()
    plt.show()

def plot_sample_comparison(dataset, sample_idx=0, figsize=(15, 10), save_path=None):
    """
    Plot a sample showing raw vs normalized data for all input and output channels.
    
    Args:
        dataset: AirfransDataset instance
        sample_idx: Index of sample to visualize
        figsize: Figure size
        save_path: Optional path to save the plots
    """
    
    # Channel names for labels
    input_channels = ['u_input', 'v_input', 'mask_binary', 'sdf_fixed']
    output_channels = ['u_target', 'v_target', 'p_target', 'nut_fixed']
    
    # Get raw sample from train_db (without normalization)
    raw_sample = dataset.train_db[sample_idx]
    x_raw = raw_sample['x']  # Shape: (C, H, W)
    y_raw = raw_sample['y']  # Shape: (C, H, W)
    
    print(f"Raw data shapes - x: {x_raw.shape}, y: {y_raw.shape}")
    
    # Debug: Print detailed statistics for each input channel
    print(f"\nDETAILED INPUT CHANNEL ANALYSIS:")
    print("=" * 50)
    for i, name in enumerate(input_channels):
        channel_data = x_raw[i]
        unique_vals = torch.unique(channel_data)
        
        # Check for NaNs and Infs
        nan_count = torch.isnan(channel_data).sum()
        inf_count = torch.isinf(channel_data).sum()
        
        print(f"{name:12s}: min={channel_data.min():.6f}, max={channel_data.max():.6f}, "
              f"mean={channel_data.mean():.6f}, std={channel_data.std():.6f}")
        print(f"{'':12s}  unique_values={len(unique_vals)} first_few={unique_vals[:5].tolist()}")
        print(f"{'':12s}  NaNs={nan_count}, Infs={inf_count}")
        
        # Special check for mask_binary (should be 0s and 1s)
        if name == 'mask_binary':
            zeros = (channel_data == 0).sum()
            ones = (channel_data == 1).sum()
            other_values = (channel_data != 0) & (channel_data != 1)
            other_count = other_values.sum()
            print(f"{'':12s}  mask: {zeros} zeros, {ones} ones, {other_count} other_values")
            if zeros == 0:
                print(f"{'':12s}  *** WARNING: NO AIRFOIL GEOMETRY IN MASK! ***")
        
        # Special check for velocity fields (should not be constant)
        if name in ['u_input', 'v_input'] and channel_data.std() < 1e-3:
            print(f"{'':12s}  *** WARNING: {name} appears constant! ***")
    print()
    
    # Get normalized sample using data_processor
    if dataset.data_processor is not None:
        # Add batch dimension for preprocessing
        batched_sample = {
            'x': x_raw.unsqueeze(0),  # Shape: (1, C, H, W)
            'y': y_raw.unsqueeze(0)   # Shape: (1, C, H, W)
        }
        processed_sample = dataset.data_processor.preprocess(batched_sample, batched=True)
        # Remove batch dimension
        x_norm = processed_sample['x'].squeeze(0)  # Shape: (C, H, W)
        y_norm = processed_sample['y'].squeeze(0)  # Shape: (C, H, W)
        print(f"Normalized data shapes - x: {x_norm.shape}, y: {y_norm.shape}")
    else:
        x_norm = x_raw
        y_norm = y_raw
        print("No data processor found - showing raw data in both columns")
    
    # Convert to numpy for plotting
    x_raw_np = x_raw.detach().numpy()
    y_raw_np = y_raw.detach().numpy()
    x_norm_np = x_norm.detach().numpy()
    y_norm_np = y_norm.detach().numpy()
    
    # Plot INPUT channels
    fig_input, axes_input = plt.subplots(4, 2, figsize=figsize)
    fig_input.suptitle(f'Input Channels - Sample {sample_idx}', fontsize=16, fontweight='bold')
    
    for i in range(4):
        # Choose appropriate colormap for each channel type
        if input_channels[i] == 'mask_binary':
            cmap = 'gray'
            vmin, vmax = 0, 1  # Force binary range
        elif input_channels[i] == 'sdf_fixed':
            cmap = 'RdBu_r'
            vmin, vmax = None, None  # Auto range
        else:  # velocity channels
            cmap = 'RdBu_r'
            vmin, vmax = None, None  # Auto range
        
        # Raw data (left column)
        im1 = axes_input[i, 0].imshow(x_raw_np[i], cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
        axes_input[i, 0].set_title(f'{input_channels[i]} (Raw)\nrange: [{x_raw_np[i].min():.3f}, {x_raw_np[i].max():.3f}]')
        axes_input[i, 0].axis('off')
        plt.colorbar(im1, ax=axes_input[i, 0], fraction=0.046, pad=0.04)
        
        # Normalized data (right column)  
        im2 = axes_input[i, 1].imshow(x_norm_np[i], cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
        axes_input[i, 1].set_title(f'{input_channels[i]} (Normalized)\nrange: [{x_norm_np[i].min():.3f}, {x_norm_np[i].max():.3f}]')
        axes_input[i, 1].axis('off')
        plt.colorbar(im2, ax=axes_input[i, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_input_channels.png", dpi=150, bbox_inches='tight')
    
    # Plot OUTPUT channels
    fig_output, axes_output = plt.subplots(4, 2, figsize=figsize)
    fig_output.suptitle(f'Output Channels - Sample {sample_idx}', fontsize=16, fontweight='bold')
    
    for i in range(4):
        # Raw data (left column)
        im1 = axes_output[i, 0].imshow(y_raw_np[i], cmap='RdBu_r', aspect='equal')
        axes_output[i, 0].set_title(f'{output_channels[i]} (Raw)')
        axes_output[i, 0].axis('off')
        plt.colorbar(im1, ax=axes_output[i, 0], fraction=0.046, pad=0.04)
        
        # Normalized data (right column)
        im2 = axes_output[i, 1].imshow(y_norm_np[i], cmap='RdBu_r', aspect='equal')
        axes_output[i, 1].set_title(f'{output_channels[i]} (Normalized)')
        axes_output[i, 1].axis('off')
        plt.colorbar(im2, ax=axes_output[i, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_output_channels.png", dpi=150, bbox_inches='tight')
    
    plt.show()
    
    # Print statistics for comparison
    print(f"\nSample {sample_idx} Statistics:")
    print("=" * 50)
    
    print("\nINPUT CHANNELS:")
    for i, name in enumerate(input_channels):
        raw_stats = f"Raw: mean={x_raw[i].mean():.4f}, std={x_raw[i].std():.4f}"
        norm_stats = f"Norm: mean={x_norm[i].mean():.4f}, std={x_norm[i].std():.4f}"
        print(f"{name:12s} | {raw_stats} | {norm_stats}")
    
    print("\nOUTPUT CHANNELS:")
    for i, name in enumerate(output_channels):
        raw_stats = f"Raw: mean={y_raw[i].mean():.4f}, std={y_raw[i].std():.4f}"
        norm_stats = f"Norm: mean={y_norm[i].mean():.4f}, std={y_norm[i].std():.4f}"
        print(f"{name:12s} | {raw_stats} | {norm_stats}")


if __name__ == "__main__":

    from tims.airfrans.airfrans_config import Default
    # Read the configuration
    from zencfg import make_config_from_cli
    import sys

    config = make_config_from_cli(Default)
    config = config.to_dict()

    data_root = "/home/timm/Projects/PIML/neuraloperator/tims/airfrans/consolidated_data"

    # Example usage
    train_loader, test_loaders, data_processor = load_airfrans_dataset(
        data_dir=config.data.data_dir,
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
    
    print("Train loader and test loaders created successfully.")
    
    # Create dataset instance for plotting
    dataset = AirfransDatasetAll(
        data_dir=config.data.data_dir,
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
    
    # Plot sample comparison
    print("\nCreating visualization plots...")
    #plot_sample_comparison(dataset, sample_idx=13, figsize=(15 , 10))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 2. Extract the DataProcessor (which now has the fitted Normalizers)
    #data_processor = dataset.data_processor.to(device)

    # 3. Setup Model from Config
    model = get_model(config).to(device)

    # 2. Add GELU and spectral convolution to the allowlist before loading
    torch.serialization.add_safe_globals([zencfg.bunch.Bunch, torch._C._nn.gelu, neuralop.layers.spectral_convolution.SpectralConv])
    model_checkpoint_path = "/home/timm/Projects/PIML/neuraloperator/tims/airfrans/checkpoints/checkpoint_epoch_260.pt"

    # Load weights safely
    checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])

    if 'data_processor' in checkpoint:
        data_processor.load_state_dict(checkpoint['data_processor'])
        print("Status: Stats updated from checkpoint!")
        
        audit_data_processor(data_processor)
    else:
        print("CRITICAL ERROR: No data_processor state found in checkpoint!")  

    plot_model_predictions(
        model=model,
        dataset=dataset,
        data_processor=data_processor,
        sample_idx=13,
        device=device
    )
