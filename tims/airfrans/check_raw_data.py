#!/usr/bin/env python
"""
Quick check to see if our data is actually normalized or already in physical units.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, ".")

def check_single_file():
    """Check a single data file to see if it's normalized."""
    
    # Check original single simulation file
    data_file = "/home/timm/Projects/PIML/Dataset/airFoil2D_SST_67.481_-1.259_5.136_2.438_18.882/airFoil2D_SST_67.481_-1.259_5.136_2.438_18.882_X2_Y2_G64x64.pt"
    
    print("=== Original Single Simulation File ===")
    try:
        data = torch.load(data_file, weights_only=False)
        
        print(f"Keys: {list(data.keys())}")
        
        if 'x' in data:
            x = data['x']
            print(f"\nInput (x) shape: {x.shape}")
            
            input_channels = ['u_input', 'v_input', 'mask_binary', 'sdf_fixed']
            for i, name in enumerate(input_channels):
                if i < x.shape[0]:
                    mean_val = x[i].mean().item()
                    std_val = x[i].std().item()
                    min_val = x[i].min().item()
                    max_val = x[i].max().item()
                    print(f"  {name:12s}: mean={mean_val:8.4f}, std={std_val:8.4f}, range=[{min_val:8.4f}, {max_val:8.4f}]")
        
        if 'y' in data:
            y = data['y']
            print(f"\nOutput (y) shape: {y.shape}")
            
            output_channels = ['u_target', 'v_target', 'p_target', 'nut_fixed']
            for i, name in enumerate(output_channels):
                if i < y.shape[0]:
                    mean_val = y[i].mean().item()
                    std_val = y[i].std().item()
                    min_val = y[i].min().item()
                    max_val = y[i].max().item()
                    print(f"  {name:12s}: mean={mean_val:8.8f}, std={std_val:8.4f}, range=[{min_val:8.4f}, {max_val:8.4f}]")
        
    except Exception as e:
        print(f"Error loading original file: {e}")
    
    # Now check consolidated file
    print("\n" + "="*50)
    print("=== Consolidated Data File ===")
    consolidated_file = "/home/timm/Projects/PIML/neuraloperator/tims/airfrans/consolidated_data/airfoil_full_test_64.pt"
    
    try:
        data = torch.load(consolidated_file, weights_only=False)
        
        print(f"Keys: {list(data.keys())}")
        
        if 'x' in data:
            x = data['x']
            print(f"\nInput (x) shape: {x.shape}")
            
            # Check first sample
            x_sample = x[0]  # First sample
            input_channels = ['u_input', 'v_input', 'mask_binary', 'sdf_fixed']
            for i, name in enumerate(input_channels):
                if i < x_sample.shape[0]:
                    mean_val = x_sample[i].mean().item()
                    std_val = x_sample[i].std().item()
                    min_val = x_sample[i].min().item()
                    max_val = x_sample[i].max().item()
                    print(f"  {name:12s}: mean={mean_val:8.4f}, std={std_val:8.4f}, range=[{min_val:8.4f}, {max_val:8.4f}]")
        
        if 'y' in data:
            y = data['y']
            print(f"\nOutput (y) shape: {y.shape}")
            
            # Check first sample
            y_sample = y[0]
            output_channels = ['u_target', 'v_target', 'p_target', 'nut_fixed']
            for i, name in enumerate(output_channels):
                if i < y_sample.shape[0]:
                    mean_val = y_sample[i].mean().item()
                    std_val = y_sample[i].std().item()
                    min_val = y_sample[i].min().item()
                    max_val = y_sample[i].max().item()
                    print(f"  {name:12s}: mean={mean_val:8.8f}, std={std_val:8.4f}, range=[{min_val:8.4f}, {max_val:8.4f}]")
        
    except Exception as e:
        print(f"Error loading consolidated file: {e}")

if __name__ == "__main__":
    check_single_file()