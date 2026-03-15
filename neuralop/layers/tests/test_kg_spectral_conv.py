import pytest
import torch
from ..kg_spectral_conv import KGSpectralConv


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_kg_spectral_conv_output_shape(dim):
    """Test that the output shape matches expectations for 1D-3D inputs."""
    in_ch, out_ch = 3, 5
    modes = (16,) * dim
    spatial = (32,) * dim

    layer = KGSpectralConv(in_ch, out_ch, n_modes=modes)
    x = torch.randn(2, in_ch, *spatial)
    y = layer(x)

    assert y.shape == (2, out_ch, *spatial)


@pytest.mark.parametrize("per_channel", [False, True])
def test_kg_spectral_conv_per_channel(per_channel):
    """Test shared vs per-channel KG parameters."""
    in_ch, out_ch = 2, 4
    layer = KGSpectralConv(
        in_ch, out_ch, n_modes=(8,), per_channel=per_channel
    )

    expected_shape = (out_ch,) if per_channel else (1,)
    assert layer.log_T.shape == expected_shape
    assert layer.log_c.shape == expected_shape
    assert layer.log_chi.shape == expected_shape

    x = torch.randn(1, in_ch, 32)
    y = layer(x)
    assert y.shape == (1, out_ch, 32)


def test_kg_spectral_conv_gradient_flow():
    """Verify that gradients flow through all KG parameters."""
    layer = KGSpectralConv(2, 2, n_modes=(16,))
    x = torch.randn(1, 2, 64, requires_grad=True)

    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert layer.log_T.grad is not None
    assert layer.log_c.grad is not None
    assert layer.log_chi.grad is not None
    assert layer.channel_weight.grad is not None
    assert x.grad is not None


def test_kg_spectral_conv_identity_at_T0():
    """When T -> 0, the KG filter is cos(0) = 1 (identity)."""
    layer = KGSpectralConv(
        1, 1, n_modes=(16,), init_T=1e-8, init_c=1.0, init_chi=1.0, bias=False,
    )
    # Set channel weight to identity
    with torch.no_grad():
        layer.channel_weight.fill_(1.0)

    x = torch.randn(1, 1, 64)
    y = layer(x)

    # With T ~ 0, cos(T * omega) ~ 1, so output ~ input
    torch.testing.assert_close(y, x, atol=1e-4, rtol=1e-4)


def test_kg_spectral_conv_no_bias():
    """Test that bias=False works correctly."""
    layer = KGSpectralConv(2, 3, n_modes=(8,), bias=False)
    assert layer.bias is None

    x = torch.randn(1, 2, 32)
    y = layer(x)
    assert y.shape == (1, 3, 32)


def test_kg_spectral_conv_complex_data():
    """Test with complex-valued input data."""
    layer = KGSpectralConv(2, 2, n_modes=(8,), complex_data=True)
    x = torch.randn(1, 2, 32, dtype=torch.cfloat)
    y = layer(x)

    assert y.shape == (1, 2, 32)
    assert y.dtype == torch.cfloat


def test_kg_spectral_conv_parameter_efficiency():
    """KG layer should have far fewer parameters than standard SpectralConv."""
    in_ch, out_ch, n_modes = 16, 16, (32,)

    kg_layer = KGSpectralConv(in_ch, out_ch, n_modes=n_modes)
    kg_params = sum(p.numel() for p in kg_layer.parameters())

    # Standard SpectralConv would have in_ch * out_ch * prod(n_modes) complex params
    # plus bias. KG has: 3 (T, c, chi) + in_ch * out_ch (channel_weight) + out_ch (bias)
    fno_spectral_params = in_ch * out_ch * n_modes[0] * 2  # *2 for complex
    assert kg_params < fno_spectral_params


def test_kg_spectral_conv_output_shape_with_resize():
    """Test output_shape parameter for resolution changes."""
    layer = KGSpectralConv(2, 2, n_modes=(16,))
    x = torch.randn(1, 2, 64)

    y = layer(x, output_shape=(128,))
    assert y.shape == (1, 2, 128)

    y = layer(x, output_shape=(32,))
    assert y.shape == (1, 2, 32)


def test_kg_spectral_conv_2d():
    """Dedicated 2D test with non-square input."""
    layer = KGSpectralConv(1, 1, n_modes=(8, 8))
    x = torch.randn(2, 1, 32, 48)
    y = layer(x)
    assert y.shape == (2, 1, 32, 48)


def test_kg_spectral_conv_repr():
    """Test that string representation works without errors."""
    layer = KGSpectralConv(2, 3, n_modes=(8, 8))
    r = repr(layer)
    assert "KGSpectralConv" in r
    assert "in_channels=2" in r
    assert "out_channels=3" in r


def test_kg_spectral_conv_n_modes_property():
    """Test dynamic n_modes update."""
    layer = KGSpectralConv(2, 2, n_modes=(16,))
    assert layer.n_modes == [16]

    layer.n_modes = (8,)
    assert layer.n_modes == [8]

    # Should still produce valid output
    x = torch.randn(1, 2, 64)
    y = layer(x)
    assert y.shape == (1, 2, 64)
