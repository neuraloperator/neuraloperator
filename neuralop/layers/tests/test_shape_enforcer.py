# MIT License
# Copyright (c) 2024 Saman Pordanesh

import pytest
import torch
import numpy as np
from neuralop.layers.shape_enforcer import ShapeEnforcer


@pytest.mark.parametrize(
    "test_id, start_dim, input_shape, output_shape, description",
    [
        # Basic functionality
        (1, 2, (2, 3, 10, 8), [6, 12], "Basic usage with 2D data - mixed resize"),
        (2, 1, (2, 3, 8, 10), [5, 8, 10], "Start dim = 1 (channel dimension)"),
        (3, 3, (2, 3, 10, 8, 6), [4, 3], "Start dim = 3 (last two dimensions)"),
        # Cropping and padding scenarios
        (4, 2, (2, 3, 20, 30), [10, 15], "Pure cropping - all dimensions smaller"),
        (5, 2, (2, 3, 10, 15), [20, 25], "Pure padding - all dimensions larger"),
        (6, 2, (2, 3, 20, 10), [10, 15], "Mixed - first dim cropped, second padded"),
        # Edge cases
        (7, 2, (2, 3, 10, 15), [10, 15], "Same shape - no changes needed"),
        (8, 2, (2, 3, 10, 15), None, "None output shape - return input unchanged"),
        (9, 2, (2, 0, 10, 15), [8, 20], "Empty tensor - preserve empty dimension"),
        (
            10,
            2,
            (2, 3, 100, 100),
            [1, 1],
            "Extreme downsizing - very large to very small",
        ),
    ],
)
def test_shape_enforcer(test_id, start_dim, input_shape, output_shape, description):
    """
    Comprehensive test for ShapeEnforcer with various scenarios.

    Tests different combinations of:
    - input shapes and dimensions
    - output shapes
    - start_dim values
    - cropping, padding, and mixed operations
    - edge cases
    """
    # Create enforcer with the specified start_dim
    enforcer = ShapeEnforcer(start_dim=start_dim)

    # Create input tensor with the specified shape
    x = torch.randn(*input_shape)

    # Apply the shape enforcer
    y = enforcer(x, output_shape=output_shape)

    # Test case specific assertions
    if output_shape is None:
        # If output_shape is None, input should be returned unchanged
        assert y.shape == x.shape
        assert torch.all(y == x)
    else:
        # Check dimensions before start_dim are preserved
        assert y.shape[:start_dim] == x.shape[:start_dim]

        # Check dimensions after start_dim match output_shape
        assert y.shape[start_dim:] == tuple(output_shape)

        # Additional assertions for specific test cases
        if test_id == 4:  # Pure cropping
            # For cropping, check that the preserved values are unchanged
            slices = [slice(None)] * start_dim
            for i, size in enumerate(output_shape):
                slices.append(slice(0, size))
            assert torch.all(y == x[tuple(slices)])

        elif test_id == 5:  # Pure padding
            # For padding, check original values are preserved and new values are zero
            # Check original values preserved

            # Create slices to get the part of y that corresponds to the original x
            orig_slices = [slice(None)] * len(y.shape)
            for i in range(start_dim, len(x.shape)):
                orig_slices[i] = slice(0, x.shape[i])

            # Check that original values are preserved
            assert torch.all(y[tuple(orig_slices)] == x)

            # Check padded areas are zero
            for i, (orig_size, new_size) in enumerate(
                zip(x.shape[start_dim:], output_shape)
            ):
                if orig_size < new_size:
                    pad_slices = [slice(None)] * len(y.shape)
                    pad_slices[start_dim + i] = slice(orig_size, None)
                    assert torch.all(y[tuple(pad_slices)] == 0)

        elif test_id == 6:  # Mixed crop/pad
            # First dimension should be cropped
            assert y.shape[start_dim] == output_shape[0]
            # Check cropped part
            assert torch.all(y[:, :, :, : x.shape[3]] == x[:, :, : output_shape[0], :])
            # Check padded part
            if x.shape[3] < output_shape[1]:
                assert torch.all(y[:, :, :, x.shape[3] :] == 0)

        elif test_id == 9:  # Empty tensor
            # Check that empty dimension is preserved
            assert y.shape[1] == 0

        elif test_id == 10:  # Extreme downsizing
            # Check that only the top-left corner is preserved
            assert torch.all(y[:, :, 0, 0] == x[:, :, 0, 0])
