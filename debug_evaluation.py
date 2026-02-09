#!/usr/bin/env python3

"""
Simple test to isolate the exact evaluation problem
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.append('/home/timm/Projects/PIML/neuraloperator')

from tims.losses.meta_losses import WeightedFieldwiseAggregatorLoss
from neuralop.losses import LpLoss

def test_evaluation_debug():
    """Debug the exact evaluation scenario"""
    
    print("=== Debug Evaluation Issue ===\n")
    
    # Exact setup from training
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
    
    # Create eval loss exactly like in training (logging=False)
    eval_loss_object = WeightedFieldwiseAggregatorLoss(target_losses, target_mappings, target_weights, logging=False)
    
    # Simulate exact evaluation scenario from debug output
    out = {'y': torch.randn(64, 4, 128, 128), 'x': torch.randn(64, 5, 128, 128)}  # Batch size 64 like evaluation
    loss_sample = {'x': torch.randn(64, 5, 128, 128), 'y': torch.randn(64, 4, 128, 128)}
    
    print("=== Simulating exact trainer evaluation call ===")
    print(f"out type: {type(out)}, keys: {list(out.keys())}")
    print(f"loss_sample keys: {list(loss_sample.keys())}")
    print(f"out['y'] shape: {out['y'].shape}")
    print(f"loss_sample['y'] shape: {loss_sample['y'].shape}")
    
    # Test the trainer's fixed call
    y_target = loss_sample.get('y', None)
    loss_kwargs = {k: v for k, v in loss_sample.items() if k != 'y'}
    
    print(f"\ny_target type: {type(y_target)}")
    print(f"loss_kwargs keys: {list(loss_kwargs.keys())}")
    
    if y_target is not None:
        print("\nCalling: eval_loss_object(out, y_target, **loss_kwargs)")
        res = eval_loss_object(out, y_target, **loss_kwargs)
    else:
        print("\nCalling: eval_loss_object(out, **loss_sample)")
        res = eval_loss_object(out, **loss_sample)
    
    print(f"Result type: {type(res)}")
    if isinstance(res, tuple):
        val_loss_out = res[0]
        print(f"Tuple result - loss: {val_loss_out.item():.6f}")
    else:
        val_loss_out = res
        print(f"Direct result - loss: {val_loss_out.item():.6f}")
    
    print(f"Final loss value: {val_loss_out.item():.6f}")
    print(f"Is zero: {val_loss_out.item() == 0.0}")
    
    # Test different batch sizes
    print(f"\n=== Testing different batch sizes ===")
    for batch_size in [1, 4, 16, 64]:
        out_test = {'y': torch.randn(batch_size, 4, 128, 128), 'x': torch.randn(batch_size, 5, 128, 128)}
        loss_sample_test = {'x': torch.randn(batch_size, 5, 128, 128), 'y': torch.randn(batch_size, 4, 128, 128)}
        
        y_test = loss_sample_test.get('y')
        loss_kwargs_test = {k: v for k, v in loss_sample_test.items() if k != 'y'}
        
        res_test = eval_loss_object(out_test, y_test, **loss_kwargs_test)
        print(f"Batch size {batch_size}: {res_test.item():.6f}")

if __name__ == "__main__":
    test_evaluation_debug()