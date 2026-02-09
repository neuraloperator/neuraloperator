#!/usr/bin/env python3

"""
Test the fixed argument passing for WeightedFieldwiseAggregatorLoss
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.append('/home/timm/Projects/PIML/neuraloperator')

from tims.losses.meta_losses import WeightedFieldwiseAggregatorLoss
from neuralop.losses import LpLoss

def test_fixed_evaluation():
    """Test that the evaluation now works with correct argument order"""
    
    print("=== Testing Fixed Evaluation ===\n")
    
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
    
    # Create eval loss (logging=False like in training)
    eval_loss = WeightedFieldwiseAggregatorLoss(target_losses, target_mappings, target_weights, logging=False)
    
    print("=== Test 1: Original broken way (y as keyword arg) ===")
    out = {'y': torch.randn(4, 4, 128, 128), 'x': torch.randn(4, 5, 128, 128)}
    loss_sample = {'x': torch.randn(4, 5, 128, 128), 'y': torch.randn(4, 4, 128, 128)}
    
    # Original broken call (y passed as keyword arg)
    print("Calling: eval_loss(out, **loss_sample)")
    result1 = eval_loss(out, **loss_sample)
    print(f"Result: {result1.item():.6f}")
    print(f"Is zero: {result1.item() == 0.0}")
    
    print("\n=== Test 2: Fixed way (y as positional arg) ===")
    # Extract y from loss_sample to pass as positional argument  
    y_target = loss_sample.get('y')
    loss_kwargs = {k: v for k, v in loss_sample.items() if k != 'y'}
    
    # Fixed call (y passed as positional arg)
    print("Calling: eval_loss(out, y_target, **loss_kwargs)")
    result2 = eval_loss(out, y_target, **loss_kwargs)
    print(f"Result: {result2.item():.6f}")
    print(f"Is zero: {result2.item() == 0.0}")
    
    print(f"\n=== Summary ===")
    print(f"Original method result: {result1.item():.6f} (should be ~0.0)")
    print(f"Fixed method result: {result2.item():.6f} (should be non-zero)")
    print(f"Fix successful: {result2.item() != 0.0 and result1.item() == 0.0}")

if __name__ == "__main__":
    test_fixed_evaluation()