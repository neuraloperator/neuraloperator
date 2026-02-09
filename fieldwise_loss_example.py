"""
Example usage of FieldwiseAggregatorLoss from neuralop.losses.meta_losses

This example demonstrates how to set up and use FieldwiseAggregatorLoss
for a multi-field neural operator that predicts multiple physical quantities.
"""

import torch
import torch.nn as nn
from neuralop.losses.meta_losses import FieldwiseAggregatorLoss

def create_fieldwise_loss_example():
    """
    Example: Neural operator that predicts velocity (u, v) and pressure (p) fields
    - Output tensor shape: [batch_size, 3, height, width] where channels 0,1,2 = u,v,p
    - We want different loss functions for velocity vs pressure fields
    """
    
    # Define individual loss functions for each field
    losses = {
        'velocity_u': nn.MSELoss(),           # MSE for u-velocity component
        'velocity_v': nn.MSELoss(),           # MSE for v-velocity component  
        'pressure': nn.L1Loss(),              # L1 loss for pressure (less sensitive to outliers)
    }
    
    # Define field mappings - how to extract each field from the output tensor
    # Assuming output shape: [batch_size, 3, height, width]
    mappings = {
        'velocity_u': (slice(None), 0, slice(None), slice(None)),  # pred[:, 0, :, :] - first channel
        'velocity_v': (slice(None), 1, slice(None), slice(None)),  # pred[:, 1, :, :] - second channel
        'pressure':   (slice(None), 2, slice(None), slice(None)),  # pred[:, 2, :, :] - third channel
    }
    
    # Create the fieldwise aggregator loss
    fieldwise_loss = FieldwiseAggregatorLoss(
        losses=losses,
        mappings=mappings,
        logging=True  # Set to True to get individual field losses for monitoring
    )
    
    return fieldwise_loss

def demonstrate_usage():
    """Demonstrate how to use the FieldwiseAggregatorLoss"""
    
    # Create the loss function
    fieldwise_loss = create_fieldwise_loss_example()
    
    # Create sample data
    batch_size, channels, height, width = 4, 3, 32, 32
    
    # Model predictions and ground truth
    pred = torch.randn(batch_size, channels, height, width)
    truth = torch.randn(batch_size, channels, height, width)
    
    # NOTE: There's a bug in the original FieldwiseAggregatorLoss implementation
    # The view(-1, 1) operation fails because it tries to reshape non-contiguous tensors
    # Let's demonstrate with a working version first
    
    print("=== Working Manual Implementation ===")
    # Manual implementation to show how it should work
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
    
    total_loss = 0.0
    for field, indices in mappings.items():
        pred_field = pred[indices].contiguous()  # Make contiguous first
        truth_field = truth[indices].contiguous()
        field_loss = losses[field](pred_field, truth_field)
        print(f"{field} loss: {field_loss.item():.4f}")
        total_loss += field_loss
    
    # Average across fields
    total_loss = total_loss / len(mappings)
    print(f"Total aggregated loss: {total_loss.item():.4f}")
    
    # Now let's try the original (which will fail due to the bug)
    print("\n=== Original Implementation (will fail) ===")
    try:
        original_loss = fieldwise_loss(pred, truth)
        print(f"Original loss: {original_loss}")
    except RuntimeError as e:
        print(f"Error with original implementation: {e}")
        print("This is because pred[indices].view(-1, 1) fails on non-contiguous tensors")
    
    return total_loss

def advanced_example():
    """
    More advanced example with different tensor shapes and custom loss functions
    """
    
    # Custom weighted MSE loss for important regions
    class WeightedMSELoss(nn.Module):
        def __init__(self, weight_factor=2.0):
            super().__init__()
            self.weight_factor = weight_factor
            
        def forward(self, pred, truth):
            # Give higher weight to larger values (important flow regions)
            weights = 1.0 + self.weight_factor * torch.abs(truth)
            return torch.mean(weights * (pred - truth) ** 2)
    
    # Example: Different loss functions for different physical quantities
    losses = {
        'temperature': WeightedMSELoss(weight_factor=1.5),  # Custom weighted loss
        'density': nn.MSELoss(),                            # Standard MSE
        'energy': nn.SmoothL1Loss(),                        # Huber loss (robust to outliers)
    }
    
    # Example with 1D spatial data: [batch_size, n_points, n_features]
    # where features are [temperature, density, energy]
    mappings = {
        'temperature': (slice(None), slice(None), 0),  # pred[:, :, 0]
        'density':     (slice(None), slice(None), 1),  # pred[:, :, 1] 
        'energy':      (slice(None), slice(None), 2),  # pred[:, :, 2]
    }
    
    # Create aggregator
    advanced_loss = FieldwiseAggregatorLoss(
        losses=losses,
        mappings=mappings,
        logging=True
    )
    
    # Test with sample data
    batch_size, n_points, n_features = 8, 64, 3
    pred = torch.randn(batch_size, n_points, n_features)
    truth = torch.randn(batch_size, n_points, n_features)
    
    loss = advanced_loss(pred, truth)
    print(f"Advanced example loss: {loss}")
    
    return advanced_loss

def physics_informed_example():
    """
    Example for physics-informed neural networks with data and physics losses
    applied to different output components
    """
    
    # Data loss for observed quantities, physics loss for derived quantities
    losses = {
        'data_field': nn.MSELoss(),           # Direct observations
        'physics_field': nn.L1Loss(),         # Physics-informed constraints
        'boundary_field': nn.MSELoss(),       # Boundary conditions
    }
    
    # Complex slicing for different regions/fields
    mappings = {
        'data_field': (slice(None), slice(0, 2), slice(None), slice(None)),     # First 2 channels
        'physics_field': (slice(None), slice(2, 4), slice(None), slice(None)),  # Next 2 channels
        'boundary_field': (slice(None), slice(4, 6), slice(None), slice(None)), # Last 2 channels
    }
    
    pinn_loss = FieldwiseAggregatorLoss(losses, mappings, logging=True)
    
    return pinn_loss

if __name__ == "__main__":
    print("=== Basic FieldwiseAggregatorLoss Example ===")
    demonstrate_usage()
    
    print("\n=== Advanced Example with Custom Losses ===")
    advanced_example()
    
    print("\n=== Physics-Informed Example ===")
    physics_loss = physics_informed_example()
    print("Physics-informed loss created successfully!")
    
    print("\n=== Key Points ===")
    print("1. Define individual loss functions for each field in a dict")
    print("2. Create mappings dict with same keys, specifying how to slice output tensor")
    print("3. Use tuple of slices for mappings to extract specific tensor regions")
    print("4. Set logging=True to track individual field losses")
    print("5. The final loss is averaged across all fields (1/N * sum of field losses)")