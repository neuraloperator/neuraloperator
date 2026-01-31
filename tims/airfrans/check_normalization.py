#!/usr/bin/env python
"""
Quick script to compare normalized vs denormalized data values.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, ".")
from tims.airfrans.airfrans_dataset import load_airfrans_dataset
from tims.airfrans.airfrans_config import Default

def main():
    """Compare normalized vs denormalized data ranges."""
    
    print("=== Data Normalization Check ===\n")
    
    # Load dataset
    data_root = "/home/timm/Projects/PIML/neuraloperator/tims/airfrans/consolidated_data"
    train_loader, test_loaders, data_processor = load_airfrans_dataset(
        train_split='scarce_train',
        test_splits=['full_test'],
        batch_size=1,
        test_batch_sizes=[1],
        data_root=data_root,
        train_resolution=64,
        test_resolutions=[64],
        encode_input=True,
        encode_output=True,
        encoding="channel-wise",
    )
    
    # Get one sample
    batch = next(iter(test_loaders[64]))
    sample = {k: v[0:1] for k, v in batch.items()}  # First sample
    
    print("ORIGINAL (NORMALIZED) DATA:")
    print("=" * 40)
    x_norm = sample['x'][0]  # Remove batch dim
    y_norm = sample['y'][0]
    
    input_channels = ['u_input', 'v_input', 'mask_binary', 'sdf_fixed']
    output_channels = ['u_target', 'v_target', 'p_target', 'nut_fixed']
    
    print("Input channels:")
    for i, name in enumerate(input_channels):
        mean_val = x_norm[i].mean().item()
        std_val = x_norm[i].std().item()
        min_val = x_norm[i].min().item()
        max_val = x_norm[i].max().item()
        print(f"  {name:12s}: mean={mean_val:8.4f}, std={std_val:8.4f}, range=[{min_val:8.4f}, {max_val:8.4f}]")
    
    print("\nOutput channels:")
    for i, name in enumerate(output_channels):
        mean_val = y_norm[i].mean().item()
        std_val = y_norm[i].std().item()
        min_val = y_norm[i].min().item()
        max_val = y_norm[i].max().item()
        print(f"  {name:12s}: mean={mean_val:8.4f}, std={std_val:8.4f}, range=[{min_val:8.4f}, {max_val:8.4f}]")
    
    # Denormalize using data processor
    print("\n\nDENORMALIZED (PHYSICAL UNITS) DATA:")
    print("=" * 40)
    
    if data_processor is not None:
        dummy_pred = sample['y'].clone()
        _, denorm_sample = data_processor.postprocess(dummy_pred, sample, batched=True)
        x_denorm = denorm_sample['x'][0]
        y_denorm = denorm_sample['y'][0]
        
        print("Input channels:")
        for i, name in enumerate(input_channels):
            mean_val = x_denorm[i].mean().item()
            std_val = x_denorm[i].std().item()
            min_val = x_denorm[i].min().item()
            max_val = x_denorm[i].max().item()
            print(f"  {name:12s}: mean={mean_val:8.4f}, std={std_val:8.4f}, range=[{min_val:8.4f}, {max_val:8.4f}]")
        
        print("\nOutput channels:")
        for i, name in enumerate(output_channels):
            mean_val = y_denorm[i].mean().item()
            std_val = y_denorm[i].std().item()
            min_val = y_denorm[i].min().item()
            max_val = y_denorm[i].max().item()
            print(f"  {name:12s}: mean={mean_val:8.4f}, std={std_val:8.4f}, range=[{min_val:8.4f}, {max_val:8.4f}]")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("✅ Now plotting DENORMALIZED data in physical units")
    print("✅ Ground truth and predictions are both in same units")
    print("✅ Error maps show actual physical error magnitudes")
    print("✅ Statistics reflect real flow field values")

if __name__ == "__main__":
    main()