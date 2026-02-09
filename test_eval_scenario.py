#!/usr/bin/env python3

import torch
import sys
import os

# Add the current directory to Python path to import local modules
sys.path.append('/home/timm/Projects/PIML/neuraloperator')

from tims.losses.meta_losses import WeightedFieldwiseAggregatorLoss
from neuralop.losses import LpLoss

def test_evaluation_scenario():
    """Test the exact evaluation scenario from the debug output"""
    
    print("=== Testing Evaluation Scenario ===\n")
    
    # Create losses and mappings exactly as in training
    target_losses = {
        'u_def': LpLoss(d=2, p=2),
        'v_def': LpLoss(d=2, p=2), 
        'cp': LpLoss(d=2, p=2),
        'lognutratio': LpLoss(d=2, p=2)
    }
    
    target_mappings = {
        'u_def': (slice(None), 0, slice(None), slice(None)),
        'v_def': (slice(None), 1, slice(None), slice(None)),
        'cp': (slice(None), 2, slice(None), slice(None)),
        'lognutratio': (slice(None), 3, slice(None), slice(None))
    }
    
    target_weights = {'u_def': 1.0, 'v_def': 1.0, 'cp': 1.0, 'lognutratio': 1.0}
    
    # Create eval loss (logging=False)
    eval_loss = WeightedFieldwiseAggregatorLoss(target_losses, target_mappings, target_weights, logging=False)
    
    # Create test data matching the debug output
    # out is a dict with keys ['y', 'x']
    # loss_sample has keys ['x', 'y']
    
    out = {
        'y': torch.randn(4, 4, 128, 128),  # 4 channels: u_def, v_def, cp, lognutratio
        'x': torch.randn(4, 5, 128, 128)   # Input
    }
    
    loss_sample = {
        'x': torch.randn(4, 5, 128, 128),  # Input
        'y': torch.randn(4, 4, 128, 128)   # Ground truth
    }
    
    print("Testing data:")
    print(f"  out['y'] shape: {out['y'].shape}")
    print(f"  loss_sample['y'] shape: {loss_sample['y'].shape}")
    
    # Call exactly like in the trainer
    print(f"\nCalling: loss_fn(out, **loss_sample)")
    res = eval_loss(out, **loss_sample)
    
    print(f"Result: {res}")
    if torch.is_tensor(res):
        print(f"Loss value: {res.item():.6f}")
    
    return res

if __name__ == "__main__":
    test_evaluation_scenario()