import pytest
import torch
import numpy as np
from ..spectral_convolution_laplace import SpectralConvLaplace

@pytest.mark.parametrize('dim', [1, 2, 3, 4])
@pytest.mark.parametrize('in_channels', [1, 4])
@pytest.mark.parametrize('out_channels', [1, 4])
@pytest.mark.parametrize('output_shape', [False, True])
def test_SpectralConvLaplace(dim, in_channels, out_channels, output_shape):
    """Test for SpectralConvLaplace of any order
    
    Verifies:
    - Forward pass completes successfully
    - Output shape is correct
    """
    modes = (8, 6, 5, 4)
    dtype = torch.float32
    batch_size = 2
    in_channels = in_channels
    out_channels = out_channels
    
    # Create the convolutional layer
    conv = SpectralConvLaplace(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=modes[:dim],
        bias=True,
    )
    
    # Input tensor
    x = torch.randn(batch_size, in_channels, *(12,)*dim, dtype=dtype)
    expected_output_shape = (batch_size, out_channels, *(6,)*dim) if output_shape else None
    
    if output_shape:
        # Forward pass
        res = conv(x, output_shape=expected_output_shape[2:])
        
        # Check output shape
        assert res.shape[0] == batch_size
        assert res.shape[1] == out_channels
        assert list(res.shape[2:]) == list(expected_output_shape[2:])
    
    else:
        # Forward pass
        res = conv(x)
        
        # Check output shape
        assert res.shape[0] == batch_size
        assert res.shape[1] == out_channels
        assert list(res.shape[2:]) == [12]*dim