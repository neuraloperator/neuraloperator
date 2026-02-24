"""Quick tests for SFNO (Spherical FNO). SFNO is FNO with SphericalConv; 2D only."""
import pytest
import torch

try:
    from neuralop.models import SFNO
except ImportError:
    SFNO = None

@pytest.mark.skipif(SFNO is None, reason="SFNO not available (torch_harmonics required)")
@pytest.mark.parametrize("n_modes", [(4, 4), (5, 5), (4, 6), (5, 6)])
@pytest.mark.parametrize(
    "implementation,factorization,resolution_scaling_factor,rank,fixed_rank_modes",
    [
        ("factorized", "dense", None, 0.3, False),
        ("reconstructed", "cp", 1.0, 0.5, False),
        ("factorized", "tucker", None, 0.5, True),
        ("reconstructed", "tt", 1.0, 0.2, True),
    ],
)
def test_sfno_forward(
    n_modes, implementation, factorization, resolution_scaling_factor, rank, fixed_rank_modes
):
    """SFNO forward over n_modes, implementation, factorization, resolution scaling, rank, fixed_rank_modes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    size = (12, 12)

    model = SFNO(
        in_channels=3,
        out_channels=1,
        hidden_channels=8,
        n_modes=n_modes,
        n_layers=2,
        implementation=implementation,
        factorization=factorization,
        resolution_scaling_factor=resolution_scaling_factor,
        rank=rank,
        fixed_rank_modes=fixed_rank_modes,
    ).to(device)
    x = torch.randn(batch_size, 3, *size, dtype=torch.float32).to(device)
    out = model(x)

    assert out.shape == (batch_size, 1, *size)

@pytest.mark.skipif(SFNO is None, reason="SFNO not available (torch_harmonics required)")
@pytest.mark.parametrize("n_modes", [(4, 4), (5, 5), (4, 6), (5, 6)])
@pytest.mark.parametrize("resolution_scaling_factor,expected_scale", [(0.5, 0.5), (2.0, 2.0)])
def test_sfno_forward_superresolution(n_modes, resolution_scaling_factor, expected_scale):
    """SFNO with single layer and resolution scaling; output spatial size scales accordingly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    size = (12, 12)

    model = SFNO(
        in_channels=3,
        out_channels=1,
        hidden_channels=8,
        n_modes=n_modes,
        n_layers=1,
        resolution_scaling_factor=resolution_scaling_factor,
    ).to(device)
    x = torch.randn(batch_size, 3, *size, dtype=torch.float32).to(device)
    out = model(x)

    expected_h = round(size[0] * expected_scale)
    expected_w = round(size[1] * expected_scale)
    assert out.shape == (batch_size, 1, expected_h, expected_w)

@pytest.mark.skipif(SFNO is None, reason="SFNO not available (torch_harmonics required)")
@pytest.mark.parametrize("n_modes", [(4, 4), (5, 5), (4, 6), (5, 6)])
@pytest.mark.parametrize("n_layers", [1, 2])
def test_sfno_backward(n_modes, n_layers):
    """SFNO backward pass runs and all parameters receive gradients."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    size = (12, 12)

    model = SFNO(
        in_channels=2,
        out_channels=1,
        hidden_channels=4,
        n_modes=n_modes,
        n_layers=n_layers,
    ).to(device)

    x = torch.randn(batch_size, 2, *size, dtype=torch.float32).to(device)
    out = model(x)
    loss = out.sum()
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None, "All parameters should receive gradients"