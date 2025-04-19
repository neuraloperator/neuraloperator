# MIT License
# Copyright (c) 2024 Saman Pordanesh

import pytest
import torch
import numpy as np
from neuralop.layers.spectral_convolution_laplace import (
    SpectralConvLaplace1D,
    SpectralConvLaplace2D,
    SpectralConvLaplace3D,
)

# ------------------------------------------------------------------
# 1D Laplace Convolution Tests
# ------------------------------------------------------------------


def test_laplace1d_forward_basic():
    """
    Basic forward test for SpectralConvLaplace1D:
      - Checks the output has correct shape
      - Verifies it returns a real tensor if input is real
      - Ensures no runtime error for standard usage
    """
    batch_size, in_channels, out_channels = 2, 3, 2
    length = 16
    n_modes = (8,)

    conv = SpectralConvLaplace1D(
        in_channels=in_channels, out_channels=out_channels, n_modes=n_modes
    )

    x = torch.randn(batch_size, in_channels, length, dtype=torch.float32)
    y = conv(x)
    assert y.shape == (batch_size, out_channels, length)
    # The module internally does ifft, and we expect real output from real input
    assert torch.is_floating_point(y)


def test_laplace1d_transform_scaling_up_down():
    """
    Test the transform() method in SpectralConvLaplace1D:
      - Verifies upsampling and downsampling
      - Ensures shape changes are correct
    """
    conv_down = SpectralConvLaplace1D(
        in_channels=1, out_channels=1, n_modes=(4,), resolution_scaling_factor=0.5
    )
    conv_up = SpectralConvLaplace1D(
        in_channels=1, out_channels=1, n_modes=(4,), resolution_scaling_factor=2.0
    )

    x = torch.randn(1, 1, 12)  # (B, C, L)

    x_down = conv_down.transform(x)
    x_up = conv_up.transform(x)

    # 12 -> 6
    assert x_down.shape == (1, 1, 6)
    # 12 -> 24
    assert x_up.shape == (1, 1, 24)


def test_laplace1d_custom_domain():
    """
    Checks that user-specified domain (start_points, end_points, linspace_steps)
    runs correctly. dt should be computed from the shape and domain.
    We only check that it runs without error and the shape is retained.
    """
    conv = SpectralConvLaplace1D(
        in_channels=1,
        out_channels=1,
        n_modes=(5,),
        linspace_steps=[20],
        linspace_startpoints=[-2.0],
        linspace_endpoints=[2.0],
    )
    x = torch.randn(1, 1, 20)
    y = conv(x)
    # shape must remain the same if transform() is not triggered:
    assert y.shape == (1, 1, 20)


def test_laplace1d_invalid_linspace():
    """
    Ensures a ValueError is raised if user-specified domain arguments
    don't match the dimension.
    For 1D, we need exactly 1 start point and 1 end point.
    """
    x = torch.rand(1, 2, 3)
    conv = SpectralConvLaplace1D(
        in_channels=2,
        out_channels=2,
        n_modes=(4,),
        linspace_steps=[16],
        linspace_startpoints=[0.0, 1.0],  # Mismatch: we have 2
        linspace_endpoints=[2.0],
    )
    with pytest.raises(ValueError):
        y = conv(x)


def test_laplace1d_pole_residue_manual_weight():
    """
    Test the core 'pole-residue' logic in 1D by setting the conv.weight
    to a known pattern and verifying partial outputs:
      - Check that transient (x1) and steady-state-like (x2) parts
        contribute differently.
      - This is a *sanity* check that output_PR is invoked as expected.
    """
    in_channels, out_channels = 1, 1
    conv = SpectralConvLaplace1D(
        in_channels=in_channels, out_channels=out_channels, n_modes=(4,)
    )
    # Overwrite the initial random weights with a small, known, complex pattern
    # conv.weight shape => (in_channels, out_channels, total_modes)
    # total_modes = max_n_modes + max_n_modes for 1D
    w_shape = conv.weight.shape
    # Example pattern: real part = 0, imag part = index
    # so conv.weight[i] = i * 1j, purely imaginary
    w_data = torch.zeros_like(conv.weight)
    for idx in range(w_shape[-1]):
        w_data[..., idx] = 1j * (idx + 1)
    conv.weight.data = w_data

    x = torch.rand(2, in_channels, 8)
    y = conv(x)

    # We can't know the exact numeric answer without writing an entire symbolic solution,
    # but we can at least confirm the shape and that the output is real.
    assert y.shape == (2, out_channels, 8)
    assert torch.is_floating_point(y)


# ------------------------------------------------------------------
# 2D Laplace Convolution Tests
# ------------------------------------------------------------------


def test_laplace2d_forward_basic():
    """
    Basic forward test for SpectralConvLaplace2D.
    We verify that the output shape matches the input shape in H/W
    and that no runtime error occurs.
    """
    batch_size, in_channels, out_channels = 2, 2, 3
    height, width = 16, 12
    conv = SpectralConvLaplace2D(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=(6, 6),
    )

    x = torch.randn(batch_size, in_channels, height, width)
    y = conv(x)
    assert y.shape == (batch_size, out_channels, height, width)
    assert torch.is_floating_point(y)


@pytest.mark.parametrize(
    "n_modes,max_n_modes",
    [
        ((10, 8), None),
        ((10, 8), (6, 4)),  # forced truncation
    ],
)
def test_laplace2d_mode_truncation(n_modes, max_n_modes):
    """
    Validate that 2D Laplace conv properly truncates modes if
    max_n_modes < n_modes or input dimension < n_modes.
    """
    conv = SpectralConvLaplace2D(
        in_channels=1, out_channels=1, n_modes=n_modes, max_n_modes=max_n_modes
    )
    # shape is 8x6, smaller than some of the modes
    x = torch.randn(1, 1, 8, 6)
    y = conv(x)
    assert y.shape == (1, 1, 8, 6)


def test_laplace2d_transform_up_down():
    """
    Test transform() in 2D with different resolution_scaling_factors.
    Ensures that shape changes are correct.
    """
    conv_down = SpectralConvLaplace2D(
        in_channels=1, out_channels=1, n_modes=(4, 4), resolution_scaling_factor=0.5
    )
    conv_up = SpectralConvLaplace2D(
        in_channels=1, out_channels=1, n_modes=(4, 4), resolution_scaling_factor=2.0
    )
    x = torch.randn(1, 1, 10, 12)

    x_down = conv_down.transform(x)
    x_up = conv_up.transform(x)

    # 10 -> 5, 12 -> 6
    assert x_down.shape == (1, 1, 5, 6)
    # 10 -> 20, 12 -> 24
    assert x_up.shape == (1, 1, 20, 24)


def test_laplace2d_pole_residue_sanity():
    """
    Test partial sanity on the 'pole-residue' approach for 2D:
    - Overwrite conv.weight to a small, controlled pattern
    - Ensure the forward pass completes and yields real output
    """
    conv = SpectralConvLaplace2D(in_channels=1, out_channels=1, n_modes=(3, 3))
    # total_modes = n_modes1 + n_modes2 + (n_modes1 * n_modes2)
    # e.g. = 3 + 3 + 3*3 = 3 + 3 + 9 = 15
    w_shape = conv.weight.shape
    assert w_shape[-1] == 15

    w_data = torch.zeros_like(conv.weight)
    # Fill with a small imaginary pattern again
    for idx in range(w_data.shape[-1]):
        w_data[..., idx] = 0.01j * (idx + 1)
    conv.weight.data = w_data

    x = torch.rand(1, 1, 8, 8)
    y = conv(x)
    assert y.shape == (1, 1, 8, 8)
    assert torch.is_floating_point(y)


# ------------------------------------------------------------------
# 3D Laplace Convolution Tests
# ------------------------------------------------------------------


def test_laplace3d_forward_basic():
    """
    Basic forward test for SpectralConvLaplace3D:
      - Checks shape consistency
      - Ensures real output for real input
    """
    batch_size, in_channels, out_channels = 1, 2, 2
    depth, height, width = 4, 5, 6
    conv = SpectralConvLaplace3D(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=(4, 3, 2),
    )

    x = torch.randn(batch_size, in_channels, depth, height, width)
    y = conv(x)
    assert y.shape == (batch_size, out_channels, depth, height, width)
    assert torch.is_floating_point(y)


def test_laplace3d_transform_scaling():
    """
    Check transform() in 3D: up/down-sampling.
    """
    conv = SpectralConvLaplace3D(
        in_channels=1,
        out_channels=1,
        n_modes=(2, 2, 2),
        resolution_scaling_factor=[2.0, 0.5, 1.5],
    )
    x = torch.randn(1, 1, 6, 6, 6)
    # 6 -> 12, 6 -> 3, 6 -> 9
    x_t = conv.transform(x)
    assert x_t.shape == (1, 1, 12, 3, 9)


@pytest.mark.parametrize(
    "n_modes,max_n_modes",
    [
        ((4, 4, 4), None),
        ((5, 5, 5), (3, 6, 5)),  # partial truncation
    ],
)
def test_laplace3d_mode_truncation(n_modes, max_n_modes):
    """
    3D mode truncation check:
      - ensures we never exceed input dims (or max_n_modes) in each dimension
    """
    conv = SpectralConvLaplace3D(
        in_channels=2, out_channels=2, n_modes=n_modes, max_n_modes=max_n_modes
    )
    x = torch.randn(2, 2, 4, 5, 6)
    y = conv(x)
    assert y.shape == (2, 2, 4, 5, 6)


def test_laplace3d_pole_residue_sanity():
    """
    Minimal sanity test for the pole-residue approach in 3D:
    - Manually set the combined weight array to a small imaginary pattern
    - Forward pass should still produce real output with correct shape
    """
    conv = SpectralConvLaplace3D(
        in_channels=1,
        out_channels=1,
        n_modes=(2, 2, 2),
    )
    w_shape = conv.weight.shape
    # total_modes = modes1 + modes2 + modes3 + (modes1*modes2*modes3)
    # e.g. = 2+2+2 + (2*2*2)=6 + 8=14
    assert w_shape[-1] == 14

    w_data = torch.zeros_like(conv.weight)
    for idx in range(w_data.shape[-1]):
        w_data[..., idx] = 0.001j * (idx + 1)
    conv.weight.data = w_data

    x = torch.rand(2, 1, 4, 4, 4)
    y = conv(x)
    assert y.shape == (2, 1, 4, 4, 4)
    assert torch.is_floating_point(y)


# ------------------------------------------------------------------
# Shared Edge Cases Across All Dimensions
# ------------------------------------------------------------------
