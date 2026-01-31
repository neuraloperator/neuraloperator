#!/usr/bin/env python
"""
Debug the normalizer and denormalization process.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, ".")
from tims.airfrans.airfrans_dataset import load_airfrans_dataset

def debug_normalizers():
    """Check what methods the normalizers have."""
    
    # Load dataset to get the normalizers
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
    
    print("=== Data Processor Info ===")
    print(f"Type: {type(data_processor)}")
    print(f"Has in_normalizer: {hasattr(data_processor, 'in_normalizer')}")
    print(f"Has out_normalizer: {hasattr(data_processor, 'out_normalizer')}")
    
    if hasattr(data_processor, 'out_normalizer') and data_processor.out_normalizer:
        out_norm = data_processor.out_normalizer
        print(f"\nOutput normalizer type: {type(out_norm)}")
        print(f"Available methods: {[m for m in dir(out_norm) if not m.startswith('_')]}")
        print(f"Has inverse_transform: {hasattr(out_norm, 'inverse_transform')}")
        print(f"Has transform: {hasattr(out_norm, 'transform')}")
        print(f"Has mean: {hasattr(out_norm, 'mean')}")
        print(f"Has std: {hasattr(out_norm, 'std')}")
        
        if hasattr(out_norm, 'mean') and out_norm.mean is not None:
            print(f"Mean shape: {out_norm.mean.shape}")
            print(f"Mean values: {out_norm.mean}")
        if hasattr(out_norm, 'std') and out_norm.std is not None:
            print(f"Std shape: {out_norm.std.shape}")  
            print(f"Std values: {out_norm.std}")
    
    # Get a sample to test
    batch = next(iter(test_loaders[64]))
    sample = {k: v[0:1] for k, v in batch.items()}  # First sample
    
    print(f"\n=== Sample Data Ranges ===")
    print(f"Input (x) range: [{sample['x'].min():.4f}, {sample['x'].max():.4f}]")
    print(f"Output (y) range: [{sample['y'].min():.4f}, {sample['y'].max():.4f}]")
    
    # Test normalizer manually
    if hasattr(data_processor, 'out_normalizer') and data_processor.out_normalizer:
        y_test = sample['y']
        print(f"\nOriginal y shape: {y_test.shape}")
        print(f"Original y range: [{y_test.min():.4f}, {y_test.max():.4f}]")
        
        # Try to use the normalizer
        try:
            if hasattr(data_processor.out_normalizer, 'inverse_transform'):
                y_denorm = data_processor.out_normalizer.inverse_transform(y_test)
                print(f"Denormalized y range: [{y_denorm.min():.4f}, {y_denorm.max():.4f}]")
            elif hasattr(data_processor.out_normalizer, '__call__'):
                print("Normalizer is callable but no inverse_transform method")
        except Exception as e:
            print(f"Error using normalizer: {e}")

if __name__ == "__main__":
    debug_normalizers()