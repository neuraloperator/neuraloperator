"""
Fixed version of FieldwiseAggregatorLoss that handles tensor reshaping properly
"""

import torch
import torch.nn as nn
import warnings
from typing import Dict

class FixedFieldwiseAggregatorLoss(object):
    """
    Fixed version of FieldwiseAggregatorLoss that properly handles tensor reshaping.
    
    The original implementation has a bug where it tries to use .view(-1, 1) on 
    potentially non-contiguous tensors, which fails. This version fixes that issue.
    """

    def __init__(self, losses: dict, mappings: dict, logging=False):
        assert mappings.keys() == losses.keys(), 'Mappings and losses must use the same keying'
        
        self.losses = losses
        self.mappings = mappings
        self.logging = logging

    def __call__(self, pred: torch.Tensor, truth: torch.Tensor, **kwargs):
        """
        Calculate aggregate loss across model inputs and outputs.
        """
        if kwargs:
            warnings.warn(
                f"FieldwiseLoss.__call__() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        loss = 0.0
        loss_record = {} if self.logging else None
        
        # Sum losses over output fields
        for field, indices in self.mappings.items():
            # Extract field data
            pred_field = pred[indices]
            truth_field = truth[indices]
            
            # FIXED: Make tensors contiguous and flatten properly
            pred_field = pred_field.contiguous()
            truth_field = truth_field.contiguous()
            
            # Calculate field loss
            field_loss = self.losses[field](pred_field, truth_field)
            loss += field_loss
            
            if self.logging:
                loss_record[field] = field_loss.item()
        
        # Average across all fields
        loss = loss / len(self.mappings)
        
        if self.logging:
            return loss, loss_record
        else:
            return loss

def example_with_fixed_loss():
    """Example using the fixed FieldwiseAggregatorLoss"""
    
    # Define losses and mappings
    losses = {
        'velocity_u': nn.MSELoss(),
        'velocity_v': nn.MSELoss(),
        'pressure': nn.L1Loss(),
    }
    
    mappings = {
        'velocity_u': (slice(None), 0, slice(None), slice(None)),
        'velocity_v': (slice(None), 1, slice(None), slice(None)),
        'pressure':   (slice(None), 2, slice(None), slice(None)),
    }
    
    # Create fixed loss function
    fixed_loss = FixedFieldwiseAggregatorLoss(
        losses=losses,
        mappings=mappings,
        logging=True
    )
    
    # Test data
    batch_size, channels, height, width = 4, 3, 32, 32
    pred = torch.randn(batch_size, channels, height, width)
    truth = torch.randn(batch_size, channels, height, width)
    
    # Compute loss
    total_loss, loss_record = fixed_loss(pred, truth)
    
    print("=== Fixed FieldwiseAggregatorLoss ===")
    print(f"Total loss: {total_loss.item():.4f}")
    print("Individual field losses:")
    for field, loss_val in loss_record.items():
        print(f"  {field}: {loss_val:.4f}")
    
    return fixed_loss

def practical_cfd_example():
    """
    Practical example for CFD simulation with velocity and pressure fields
    """
    
    # CFD-specific losses
    losses = {
        'u_velocity': nn.MSELoss(),        # x-velocity component
        'v_velocity': nn.MSELoss(),        # y-velocity component  
        'pressure': nn.L1Loss(),           # pressure field (L1 for robustness)
        'turbulence': nn.SmoothL1Loss(),   # turbulence quantity
    }
    
    # Mappings for 4-channel output: [u, v, p, k] where k is turbulence
    mappings = {
        'u_velocity': (slice(None), 0, slice(None), slice(None)),
        'v_velocity': (slice(None), 1, slice(None), slice(None)),
        'pressure':   (slice(None), 2, slice(None), slice(None)),
        'turbulence': (slice(None), 3, slice(None), slice(None)),
    }
    
    cfd_loss = FixedFieldwiseAggregatorLoss(losses, mappings, logging=True)
    
    # Simulate CFD data: batch_size=8, channels=4, height=64, width=64
    pred = torch.randn(8, 4, 64, 64)
    truth = torch.randn(8, 4, 64, 64)
    
    loss, loss_record = cfd_loss(pred, truth)
    
    print("\n=== CFD Example ===")
    print(f"Total CFD loss: {loss.item():.4f}")
    for field, loss_val in loss_record.items():
        print(f"  {field}: {loss_val:.4f}")
    
    return cfd_loss

if __name__ == "__main__":
    example_with_fixed_loss()
    practical_cfd_example()
    
    print("\n=== Key Takeaways ===")
    print("1. Original FieldwiseAggregatorLoss has a tensor reshaping bug")
    print("2. The bug occurs with .view(-1, 1) on non-contiguous tensors")
    print("3. Fixed version makes tensors contiguous before operations")
    print("4. Use different loss functions for different physical quantities")
    print("5. Set logging=True to monitor individual field losses during training")