#!/usr/bin/env python3

import torch

def test_tensor_formatting():
    """Test how the log_eval tensor formatting works"""
    
    # Test different tensor types
    scalar_tensor = torch.tensor(5.6542)
    print(f"Scalar tensor: {scalar_tensor:.4f}")
    
    # Test what happens with non-scalar tensors
    try:
        multi_dim_tensor = torch.tensor([5.6542, 3.2341])
        print(f"Multi-dim tensor: {multi_dim_tensor:.4f}")
    except Exception as e:
        print(f"Multi-dim tensor error: {e}")
    
    # Test 0-dimensional tensor
    zero_dim = torch.tensor(0.0)
    print(f"Zero tensor: {zero_dim:.4f}")
    
    # Test tensor with .item()
    try:
        multi_dim_tensor = torch.tensor([5.6542, 3.2341])
        print(f"Multi-dim with .item() would fail: {multi_dim_tensor.item():.4f}")
    except Exception as e:
        print(f"Multi-dim .item() error: {e}")

if __name__ == "__main__":
    test_tensor_formatting()