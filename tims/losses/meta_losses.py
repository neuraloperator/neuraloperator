import torch
import torch.nn as nn
import logging
from torch import nn
from typing import Dict, List, Optional, Callable


class WeightedFieldwiseAggregatorLoss(object):
    """
    Weighted aggregator that slices multi-physics output channels (u, v, p, nut)
    and applies specific loss functions and weights to each, preserving 
    spatial dimensions for quadrature-based losses like LpLoss.
    """
    def __init__(self, losses: dict, mappings: dict, weights: dict = None, logging=True):
        super().__init__()
        # Ensure we have a loss and a mapping for every field
        assert mappings.keys() == losses.keys(), "Mappings and losses must share keys."
        
        self.losses = losses # ModuleDict handles device placement
        self.mappings = mappings
        self.logging = logging
        
        # Default to equal weighting if none provided
        if weights is None:
            self.weights = {k: 1.0 / len(losses) for k in losses.keys()}
        else:
            self.weights = weights
        # Sum of weights for normalization 
        self.weight_sum = sum(self.weights.values())

    def forward(self, pred, y=None, **kwargs):

        # 1. If y is None, try to get it from kwargs (fallback)
        if y is None:
            y = kwargs.get('y')
            #print(f"⚠️ MetaLoss: y was None, got from kwargs: {type(y)}")

        # 2. Robust unwrapping for pred
        if isinstance(pred, dict):
            # Try 'y', then 'out', then the first value in the dict
            pred = pred.get('y', pred.get('out', next(iter(pred.values()))))

        # 3. Robust unwrapping for y (in case it's still a dict)
        if isinstance(y, dict):
            y = y.get('y', next(iter(y.values())))
        
        # 4. COMPREHENSIVE SAFETY CHECK with detailed debugging
        if pred is None or y is None or not torch.is_tensor(pred) or not torch.is_tensor(y):
            #print(f"⚠️ MetaLoss CRITICAL Error:")
            #print(f"  pred: type={type(pred)}, is_tensor={torch.is_tensor(pred) if pred is not None else 'N/A'}")
            #print(f"  y: type={type(y)}, is_tensor={torch.is_tensor(y) if y is not None else 'N/A'}")
            #print(f"  kwargs keys: {list(kwargs.keys())}")
            #print(f"⚠️ FIXED: Now y should be passed as positional argument!")
            
            zero_loss = torch.tensor(0.0, requires_grad=True)
            if hasattr(pred, 'device') and pred is not None:
                zero_loss = zero_loss.to(pred.device)
            elif hasattr(y, 'device') and y is not None:
                zero_loss = zero_loss.to(y.device)
            else:
                zero_loss = zero_loss.to('cuda')  # Fallback
                
            if self.logging:
                return zero_loss, {}
            else:
                return zero_los


        total_loss = 0.0
        loss_record = {}

        for field, indices in self.mappings.items():
            # Check if indices actually exist for this resolution

            #print(f"DEBUG: Processing field '{field}' with indices {indices}")
            p_field = pred[indices]
            t_field = y[indices]
            
            field_loss = self.losses[field](p_field, t_field)
            total_loss += (self.weights[field] / self.weight_sum) * field_loss
            
            if self.logging:
                # Standardizing key names for the Trainer's CSV/WandB logger
                # 'u_def' becomes 'enc_u_loss', etc.
                clean_name = field.replace('_def', '')
                
                loss_record[f'enc_{clean_name}_loss'] = field_loss.detach().item()
        if self.logging:
            return total_loss, loss_record
        else:
            return total_loss
    
    # 3. Explicitly define __call__ to ensure Python knows this object 
    # can be used as a function in your trainer.
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)