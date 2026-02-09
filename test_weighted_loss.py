#!/usr/bin/env python3

import torch
import sys
import os

# Add the current directory to Python path to import local modules
sys.path.append('/home/timm/Projects/PIML/neuraloperator')

from tims.losses.meta_losses import WeightedFieldwiseAggregatorLoss
from neuralop.losses import LpLoss

def test_weighted_fieldwise_loss():
    """Test the WeightedFieldwiseAggregatorLoss with sample data"""
    
    print("=== Testing WeightedFieldwiseAggregatorLoss ===\n")
    
    # Create sample losses and mappings like in your training
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
    
    target_weights = {
        'u_def': 1.0,
        'v_def': 1.0,
        'cp': 1.0,
        'lognutratio': 1.0
    }
    
    # Create loss functions
    train_loss = WeightedFieldwiseAggregatorLoss(target_losses, target_mappings, target_weights, logging=True)
    eval_loss = WeightedFieldwiseAggregatorLoss(target_losses, target_mappings, target_weights, logging=False)
    
    # Create sample data (batch_size=2, channels=4, height=64, width=64)
    pred = torch.randn(2, 4, 64, 64)
    y = torch.randn(2, 4, 64, 64)
    
    print("Testing with direct arguments:")
    print("pred shape:", pred.shape)
    print("y shape:", y.shape)
    
    # Test training loss (logging=True, should return tuple)
    print("\n--- Training Loss (logging=True) ---")
    train_result = train_loss(pred, y)
    print(f"Train result type: {type(train_result)}")
    if isinstance(train_result, tuple):
        loss_val, loss_dict = train_result
        print(f"Loss value: {loss_val.item():.6f}")
        print(f"Loss dict: {loss_dict}")
    else:
        print(f"Loss value: {train_result.item():.6f}")
    
    # Test eval loss (logging=False, should return scalar)
    print("\n--- Eval Loss (logging=False) ---")
    eval_result = eval_loss(pred, y)
    print(f"Eval result type: {type(eval_result)}")
    if isinstance(eval_result, tuple):
        loss_val, loss_dict = eval_result
        print(f"Loss value: {loss_val.item():.6f}")
        print(f"Loss dict: {loss_dict}")
    else:
        print(f"Loss value: {eval_result.item():.6f}")
    
    # Test with kwargs (as done in trainer)
    print("\n--- Testing with kwargs (like trainer) ---")
    sample_dict = {'y': y, 'props': torch.tensor([1.0, 2.0])}
    
    eval_result_kwargs = eval_loss(pred, **sample_dict)
    print(f"Eval with kwargs result type: {type(eval_result_kwargs)}")
    if isinstance(eval_result_kwargs, tuple):
        loss_val, loss_dict = eval_result_kwargs
        print(f"Loss value: {loss_val.item():.6f}")
        print(f"Loss dict: {loss_dict}")
    else:
        print(f"Loss value: {eval_result_kwargs.item():.6f}")

if __name__ == "__main__":
    test_weighted_fieldwise_loss()