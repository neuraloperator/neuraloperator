"""
Comprehensive Guide to FieldwiseAggregatorLoss Setup and Usage

This guide provides practical examples for setting up FieldwiseAggregatorLoss
for neural operators, including common pitfalls and solutions.
"""

import torch
import torch.nn as nn
from neuralop.losses.meta_losses import FieldwiseAggregatorLoss

def setup_basic_fieldwise_loss():
    """
    Basic setup for a neural operator predicting multiple physical fields.
    
    Example: Fluid dynamics with velocity (u, v) and pressure (p) fields
    Output tensor shape: [batch, 3, height, width]
    """
    
    # Step 1: Define loss functions for each field
    losses = {
        'velocity_u': nn.MSELoss(),     # u-velocity component
        'velocity_v': nn.MSELoss(),     # v-velocity component
        'pressure': nn.L1Loss(),        # pressure (L1 for robustness to outliers)
    }
    
    # Step 2: Define how to extract each field from output tensor
    # Key point: Use tuples of slices to specify tensor indexing
    mappings = {
        'velocity_u': (slice(None), 0, slice(None), slice(None)),  # [:, 0, :, :]
        'velocity_v': (slice(None), 1, slice(None), slice(None)),  # [:, 1, :, :]
        'pressure':   (slice(None), 2, slice(None), slice(None)),  # [:, 2, :, :]
    }
    
    # Step 3: Create the aggregator
    fieldwise_loss = FieldwiseAggregatorLoss(
        losses=losses,
        mappings=mappings,
        logging=True  # Enable to track individual field losses
    )
    
    return fieldwise_loss, losses, mappings

def setup_3d_fields():
    """
    Setup for 3D fields (common in computational physics)
    Output shape: [batch, channels, depth, height, width]
    """
    
    losses = {
        'temperature': nn.MSELoss(),
        'density': nn.MSELoss(),
        'velocity_x': nn.MSELoss(),
        'velocity_y': nn.MSELoss(),
        'velocity_z': nn.MSELoss(),
    }
    
    # 3D field mappings
    mappings = {
        'temperature': (slice(None), 0, slice(None), slice(None), slice(None)),
        'density':     (slice(None), 1, slice(None), slice(None), slice(None)),
        'velocity_x':  (slice(None), 2, slice(None), slice(None), slice(None)),
        'velocity_y':  (slice(None), 3, slice(None), slice(None), slice(None)),
        'velocity_z':  (slice(None), 4, slice(None), slice(None), slice(None)),
    }
    
    return FieldwiseAggregatorLoss(losses, mappings, logging=True)

def setup_physics_informed_loss():
    """
    Setup for Physics-Informed Neural Networks (PINNs)
    Different loss types for different physical constraints
    """
    
    losses = {
        'data_loss': nn.MSELoss(),           # Fit to observed data
        'pde_residual': nn.MSELoss(),        # Physics equations residual
        'boundary_condition': nn.MSELoss(),   # Boundary conditions
        'initial_condition': nn.MSELoss(),    # Initial conditions
    }
    
    # Example: Output contains stacked physics terms
    # Shape: [batch, 4, spatial_dims...]
    mappings = {
        'data_loss':         (slice(None), 0, slice(None), slice(None)),
        'pde_residual':      (slice(None), 1, slice(None), slice(None)),
        'boundary_condition': (slice(None), 2, slice(None), slice(None)),
        'initial_condition': (slice(None), 3, slice(None), slice(None)),
    }
    
    return FieldwiseAggregatorLoss(losses, mappings, logging=True)

def setup_weighted_fields():
    """
    Setup with weighted importance for different fields
    Uses WeightedSumLoss within FieldwiseAggregatorLoss
    """
    from neuralop.losses.meta_losses import WeightedSumLoss
    
    # Create weighted losses for important fields
    velocity_losses = [nn.MSELoss(), nn.L1Loss()]
    velocity_weights = [0.7, 0.3]  # Emphasize MSE but include L1
    
    losses = {
        'velocity': WeightedSumLoss(velocity_losses, velocity_weights),
        'pressure': nn.L1Loss(),
        'temperature': nn.MSELoss(),
    }
    
    mappings = {
        'velocity':    (slice(None), slice(0, 2), slice(None), slice(None)),  # First 2 channels
        'pressure':    (slice(None), 2, slice(None), slice(None)),            # Channel 2
        'temperature': (slice(None), 3, slice(None), slice(None)),            # Channel 3
    }
    
    return FieldwiseAggregatorLoss(losses, mappings, logging=True)

def demonstrate_common_pitfalls():
    """
    Common mistakes and how to avoid them
    """
    
    print("=== Common Pitfalls and Solutions ===\n")
    
    # Pitfall 1: Tensor reshaping error
    print("1. Tensor Reshaping Error:")
    print("   Problem: 'view size is not compatible with input tensor's size and stride'")
    print("   Cause: The original implementation uses .view(-1, 1) on non-contiguous tensors")
    print("   Solution: Use the fixed version or ensure tensors are contiguous\n")
    
    # Pitfall 2: Mismatched keys
    print("2. Mismatched Keys:")
    print("   Problem: AssertionError about mappings and losses keys")
    print("   Cause: Keys in losses dict don't match keys in mappings dict")
    print("   Solution: Ensure exact same keys in both dictionaries\n")
    
    # Pitfall 3: Incorrect slicing
    print("3. Incorrect Tensor Slicing:")
    print("   Problem: IndexError or unexpected tensor shapes")
    print("   Cause: Wrong slice indices for tensor dimensions")
    print("   Solution: Double-check your tensor shapes and slice indices\n")
    
    # Pitfall 4: Loss function compatibility
    print("4. Loss Function Compatibility:")
    print("   Problem: Loss functions expecting different input shapes")
    print("   Cause: Some losses need specific input shapes (e.g., classification vs regression)")
    print("   Solution: Ensure all loss functions work with your tensor shapes\n")

def training_loop_example():
    """
    Example of how to use FieldwiseAggregatorLoss in a training loop
    """
    
    # Setup
    fieldwise_loss, _, _ = setup_basic_fieldwise_loss()
    
    # Mock model and data
    class MockModel(nn.Module):
        def forward(self, x):
            # Returns [batch, 3, height, width] for u, v, p fields
            return torch.randn(x.size(0), 3, 32, 32)
    
    model = MockModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("=== Training Loop Example ===\n")
    
    # Training loop
    for epoch in range(3):
        # Mock input and target
        x = torch.randn(4, 1, 32, 32)  # Input
        target = torch.randn(4, 3, 32, 32)  # Target with 3 fields
        
        # Forward pass
        pred = model(x)
        
        # Compute fieldwise loss
        total_loss, loss_record = fieldwise_loss(pred, target)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Logging
        print(f"Epoch {epoch+1}:")
        print(f"  Total Loss: {total_loss.item():.4f}")
        for field, field_loss in loss_record.items():
            print(f"  {field}: {field_loss:.4f}")
        print()

def best_practices():
    """
    Best practices for using FieldwiseAggregatorLoss
    """
    
    print("=== Best Practices ===\n")
    
    print("1. Loss Function Selection:")
    print("   - Use MSE for smooth fields (velocity, temperature)")
    print("   - Use L1 for fields with outliers (pressure, density)")
    print("   - Use SmoothL1 for robust regression")
    print("   - Use custom losses for domain-specific requirements\n")
    
    print("2. Field Mapping:")
    print("   - Double-check tensor dimensions and channel ordering")
    print("   - Use descriptive field names")
    print("   - Test mappings with small tensors first\n")
    
    print("3. Monitoring:")
    print("   - Always enable logging=True during development")
    print("   - Track individual field losses to identify issues")
    print("   - Monitor loss balance between fields\n")
    
    print("4. Debugging:")
    print("   - Print tensor shapes during development")
    print("   - Test with simple synthetic data first")
    print("   - Check that loss decreases for each field individually\n")

if __name__ == "__main__":
    print("FieldwiseAggregatorLoss Setup Guide")
    print("=" * 40 + "\n")
    
    # Basic setup
    print("=== Basic Setup ===")
    fieldwise_loss, losses, mappings = setup_basic_fieldwise_loss()
    print(f"Created fieldwise loss with fields: {list(losses.keys())}")
    print(f"Mappings: {mappings}\n")
    
    # Demonstrate usage
    demonstrate_common_pitfalls()
    training_loop_example()
    best_practices()
    
    print("=== Summary ===")
    print("FieldwiseAggregatorLoss is powerful for multi-field neural operators")
    print("Key steps: 1) Define losses, 2) Define mappings, 3) Create aggregator")
    print("Always test with simple data first and enable logging for development")