import numpy as np
import pytest
import torch
from torch.testing import assert_close

from neuralop.losses.differentiation import FourierDiff

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
    X, Y = torch.meshgrid(x, y, indexing="ij")

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


@pytest.mark.parametrize("resolution", [(16, 16), (16, 24), (24, 16)])
@pytest.mark.parametrize("constraint_modes", [(16, 24), (8, 12)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spectral_projection_preserves_mean(resolution, constraint_modes, dtype):
    """Constant velocity fields have zero divergence and must retain their mean."""
    height, width = resolution
    mean_vec = torch.tensor([[[[1.5]], [[-0.75]]]], dtype=dtype)
    u_const = mean_vec.expand(2, 2, height, width).clone()

    u_proj = spectral_projection_divergence_free(u_const, (2 * np.pi, 3 * np.pi), constraint_modes)
    atol = 1e-13 if dtype == torch.float64 else 1e-5
    assert_close(u_proj, u_const, atol=atol, rtol=0)

    # Adding a constant offset to an arbitrary field must shift the projected mean identically
    torch.manual_seed(0)
    u_var = torch.randn(2, 2, height, width, dtype=dtype)
    u_var_proj = spectral_projection_divergence_free(u_var, (2 * np.pi, 3 * np.pi), constraint_modes)
    u_shifted_proj = spectral_projection_divergence_free(
        u_var + u_const, (2 * np.pi, 3 * np.pi), constraint_modes
    )
    assert_close(u_shifted_proj, u_var_proj + u_const, atol=atol, rtol=atol)


@pytest.mark.parametrize("resolution", [(32, 32), (32, 48), (48, 32)])
@pytest.mark.parametrize(
    "domain_size",
    [2 * np.pi, (2 * np.pi, 4 * np.pi), (3 * np.pi, 2 * np.pi)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spectral_projection_axis_consistency_and_divergence(resolution, domain_size, dtype):
    """Verify shape-consistent divergence removal and invariance of divergence-free fields."""
    height, width = resolution
    domain_h, domain_w = (
        (float(domain_size), float(domain_size))
        if isinstance(domain_size, (int, float))
        else (float(domain_size[0]), float(domain_size[1]))
    )

    xs = torch.arange(height, dtype=dtype) * (domain_h / height)
    ys = torch.arange(width, dtype=dtype) * (domain_w / width)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")

    # Divergence-free periodic stream-function flow:
    # psi = sin(2*pi*X/L_h) * cos(2*pi*Y/L_w)  =>  u_0 = dpsi/dy, u_1 = -dpsi/dx
    kx = 2 * np.pi / domain_h
    ky = 2 * np.pi / domain_w
    u_divfree = torch.stack(
        [
            -ky * torch.sin(kx * X) * torch.sin(ky * Y) + 0.5,
            -kx * torch.cos(kx * X) * torch.cos(ky * Y) - 0.25,
        ],
        dim=0,
    ).unsqueeze(0)

    atol = 1e-11 if dtype == torch.float64 else 1e-4
    u_divfree_proj = spectral_projection_divergence_free(u_divfree, domain_size, resolution)
    assert_close(u_divfree_proj, u_divfree, atol=atol, rtol=atol)

    # Arbitrary periodic field should become divergence-free and idempotent after projection
    u_general = torch.stack(
        [
            torch.sin(2 * kx * X) + 0.5 * torch.cos(ky * Y) + 1.2,
            torch.cos(kx * X) + 0.3 * torch.sin(3 * ky * Y) - 0.4,
        ],
        dim=0,
    ).unsqueeze(0)
    u_proj = spectral_projection_divergence_free(u_general, domain_size, resolution)
    u_proj_twice = spectral_projection_divergence_free(u_proj, domain_size, resolution)
    assert_close(u_proj_twice, u_proj, atol=atol, rtol=atol)

    fourier_diff = FourierDiff(dim=2, L=(domain_h, domain_w), use_fc=False)
    div = fourier_diff.divergence(u_proj)
    assert torch.max(torch.abs(div)).item() < (1e-10 if dtype == torch.float64 else 1e-4)


def test_spectral_projection_autograd_and_validation():
    """Gradients through zero-mode masking must stay finite and invalid domain_size raises."""
    u = torch.randn(2, 2, 16, 24, dtype=torch.float64, requires_grad=True)
    out = spectral_projection_divergence_free(u, (2 * np.pi, 3 * np.pi), (12, 16))
    loss = (out**2).sum()
    loss.backward()
    assert u.grad is not None
    assert torch.isfinite(u.grad).all()

    with pytest.raises(ValueError, match="domain_size"):
        spectral_projection_divergence_free(u.detach(), (1.0, 2.0, 3.0), (16, 24))

