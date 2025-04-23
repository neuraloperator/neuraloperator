import pytest
import torch
import numpy as np
from neuralop.layers.shape_enforcer import ShapeEnforcer


@pytest.mark.parametrize(
    "start_dim, input_shape, output_shape",
    [
        (2, (2, 3, 10, 8), [6, 12]),    # 2D data - mixed resize
        (1, (2, 3, 8, 10), [5, 8, 10]), # Start at channel dimension
        (3, (2, 3, 10, 8, 6), [4, 3]),  # Start at last two dimensions
    ],
)
def test_basic_functionality(start_dim, input_shape, output_shape):
    """Test basic ShapeEnforcer functionality with different start dimensions."""
    enforcer = ShapeEnforcer(start_dim=start_dim)
    x = torch.randn(*input_shape)
    y = enforcer(x, output_shape=output_shape)
    
    # Check dimensions before start_dim are preserved
    assert y.shape[:start_dim] == x.shape[:start_dim]
    # Check dimensions after start_dim match output_shape
    assert y.shape[start_dim:] == tuple(output_shape)


@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        ((2, 3, 20, 30), [10, 15]),     # All dimensions smaller
        ((2, 3, 100, 100), [1, 1]),     # Extreme downsizing
    ],
)
def test_cropping(input_shape, output_shape):
    """Test ShapeEnforcer when cropping (reducing dimensions)."""
    start_dim = 2  # Use consistent start_dim for these tests
    enforcer = ShapeEnforcer(start_dim=start_dim)
    x = torch.randn(*input_shape)
    y = enforcer(x, output_shape=output_shape)
    
    # Check dimensions
    assert y.shape[:start_dim] == x.shape[:start_dim]
    assert y.shape[start_dim:] == tuple(output_shape)
    
    # Check that the preserved values are unchanged
    slices = [slice(None)] * start_dim
    for i, size in enumerate(output_shape):
        slices.append(slice(0, size))
    assert torch.all(y == x[tuple(slices)])


@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        ((2, 3, 10, 15), [20, 25]),  # All dimensions larger
    ],
)
def test_padding(input_shape, output_shape):
    """Test ShapeEnforcer when padding (increasing dimensions)."""
    start_dim = 2  # Use consistent start_dim for these tests
    enforcer = ShapeEnforcer(start_dim=start_dim)
    x = torch.randn(*input_shape)
    y = enforcer(x, output_shape=output_shape)
    
    # Check dimensions
    assert y.shape[:start_dim] == x.shape[:start_dim]
    assert y.shape[start_dim:] == tuple(output_shape)
    
    # Create slices to get the part of y that corresponds to the original x
    orig_slices = [slice(None)] * len(y.shape)
    for i in range(start_dim, len(x.shape)):
        orig_slices[i] = slice(0, x.shape[i])
    
    # Check that original values are preserved
    assert torch.all(y[tuple(orig_slices)] == x)
    
    # Check padded areas are zero
    for i, (orig_size, new_size) in enumerate(zip(x.shape[start_dim:], output_shape)):
        if orig_size < new_size:
            pad_slices = [slice(None)] * len(y.shape)
            pad_slices[start_dim + i] = slice(orig_size, None)
            assert torch.all(y[tuple(pad_slices)] == 0)


@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        ((2, 3, 20, 10), [10, 15]),  # First dim cropped, second padded
    ],
)
def test_mixed_resize(input_shape, output_shape):
    """Test ShapeEnforcer with mixed cropping and padding."""
    start_dim = 2
    enforcer = ShapeEnforcer(start_dim=start_dim)
    x = torch.randn(*input_shape)
    y = enforcer(x, output_shape=output_shape)
    
    # Check dimensions
    assert y.shape[:start_dim] == x.shape[:start_dim]
    assert y.shape[start_dim:] == tuple(output_shape)
    
    # First dimension should be cropped
    assert y.shape[start_dim] == output_shape[0]
    
    # Check cropped part
    assert torch.all(y[:, :, :, :x.shape[3]] == x[:, :, :output_shape[0], :])
    
    # Check padded part
    if x.shape[3] < output_shape[1]:
        assert torch.all(y[:, :, :, x.shape[3]:] == 0)


@pytest.mark.parametrize(
    "start_dim, input_shape, output_shape, description",
    [
        (2, (2, 3, 10, 15), [10, 15], "Same shape - no changes needed"),
        (2, (2, 3, 10, 15), None, "None output shape - return input unchanged"),
        (2, (2, 0, 10, 15), [8, 20], "Empty tensor - preserve empty dimension"),
    ],
)
def test_edge_cases(start_dim, input_shape, output_shape, description):
    """Test ShapeEnforcer edge cases."""
    enforcer = ShapeEnforcer(start_dim=start_dim)
    x = torch.randn(*input_shape)
    y = enforcer(x, output_shape=output_shape)
    
    if output_shape is None:
        # If output_shape is None, input should be returned unchanged
        assert y.shape == x.shape
        assert torch.all(y == x)
    else:
        # Check dimensions
        assert y.shape[:start_dim] == x.shape[:start_dim]
        assert y.shape[start_dim:] == tuple(output_shape)
        
        if "Empty tensor" in description:
            # Check that empty dimension is preserved
            assert y.shape[1] == 0
