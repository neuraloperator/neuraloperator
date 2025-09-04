import torch
import pytest
import numpy as np
from ..spectral_projection import spectral_projection_divergence_free

@pytest.mark.parametrize("resolution", [(64, 64), (80, 60), (60, 80)])
@pytest.mark.parametrize("constraint_modes", [(64, 64), (50, 64), (72, 60)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spectral_projection(resolution, constraint_modes, dtype):
    """Test spectral projection functionality with different resolutions, constraint modes and dtypes."""
    
    height, width = resolution
    batch_size = 3
    domain_size = 2 * np.pi
    
    # Create a simple velocity field with some divergence
    x = torch.linspace(0, domain_size, width)
    y = torch.linspace(0, domain_size, height)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Example u_x = sin(y), u_y = cos(x)
    u_x = torch.sin(Y.T).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, height, width)
    u_y = torch.cos(X.T).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, height, width)
    u = torch.cat([u_x, u_y], dim=1).to(dtype)
    
    # Apply spectral projection
    u_projected = spectral_projection_divergence_free(u, domain_size, constraint_modes)
    
    # Check output shape and properties
    assert u_projected.shape == u.shape
    assert u_projected.dtype == dtype
    assert u_projected.device == u.device
    
    # Check no NaN or inf values
    assert not torch.isnan(u_projected).any()
    assert not torch.isinf(u_projected).any()
