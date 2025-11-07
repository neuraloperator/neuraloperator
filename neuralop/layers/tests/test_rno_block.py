import pytest
import torch
from ..rno_block import RNOBlock


@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("complex_data", [False, True])
def test_RNOBlock_basic(n_dim, complex_data):
    """Test RNOBlock with basic functionality"""
    modes = (8, 8, 8)
    width = 16
    size = [12] * n_dim
    batch_size = 2
    num_timesteps = 3

    layer = RNOBlock(
        n_modes=modes[:n_dim],
        hidden_channels=width,
        complex_data=complex_data,
    )

    dtype = torch.cfloat if complex_data else torch.float32
    x = torch.randn(batch_size, num_timesteps, width, *size, dtype=dtype)

    # Test forward pass
    out = layer(x)
    assert out.shape == (batch_size, width, *size)
    assert out.dtype == dtype

    # Test with custom hidden state
    h_init = torch.randn(batch_size, width, *size, dtype=dtype)
    out_custom = layer(x, h_init)
    assert out_custom.shape == (batch_size, width, *size)
    assert out_custom.dtype == dtype


@pytest.mark.parametrize("return_sequences", [False, True])
@pytest.mark.parametrize("complex_data", [False, True])
def test_RNOBlock_return_sequences(return_sequences, complex_data):
    """Test RNOBlock with return_sequences option"""
    modes = (8, 8)
    width = 16
    size = [12, 12]
    batch_size = 2
    num_timesteps = 3

    layer = RNOBlock(
        n_modes=modes,
        hidden_channels=width,
        return_sequences=return_sequences,
        complex_data=complex_data,
    )

    dtype = torch.cfloat if complex_data else torch.float32
    x = torch.randn(batch_size, num_timesteps, width, *size, dtype=dtype)

    out = layer(x)

    if return_sequences:
        assert out.shape == (batch_size, num_timesteps, width, *size)
    else:
        assert out.shape == (batch_size, width, *size)

    assert out.dtype == dtype


@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("complex_data", [False])  # Skip complex for resolution scaling (interpolation not supported)
def test_RNOBlock_resolution_scaling(n_dim, complex_data):
    """Test RNOBlock with resolution scaling"""
    modes = (8, 8, 8)
    width = 16
    size = [12] * n_dim
    batch_size = 2
    num_timesteps = 3

    # Test upsampling
    layer_up = RNOBlock(
        n_modes=modes[:n_dim],
        hidden_channels=width,
        resolution_scaling_factor=2.0,
        complex_data=complex_data,
    )

    dtype = torch.cfloat if complex_data else torch.float32
    x = torch.randn(batch_size, num_timesteps, width, *size, dtype=dtype)

    out_up = layer_up(x)
    expected_size_up = tuple(s * 2 for s in size)
    assert out_up.shape == (batch_size, width, *expected_size_up)
    assert out_up.dtype == dtype

    # Test downsampling
    layer_down = RNOBlock(
        n_modes=modes[:n_dim],
        hidden_channels=width,
        resolution_scaling_factor=0.5,
        complex_data=complex_data,
    )

    out_down = layer_down(x)
    expected_size_down = tuple(s // 2 for s in size)
    assert out_down.shape == (batch_size, width, *expected_size_down)
    assert out_down.dtype == dtype


@pytest.mark.parametrize("norm", [None, "group_norm", "instance_norm"])
@pytest.mark.parametrize("complex_data", [False, True])
def test_RNOBlock_norm(norm, complex_data):
    """Test RNOBlock with different normalization options"""
    modes = (8, 8)
    width = 16
    size = [12, 12]
    batch_size = 2
    num_timesteps = 3

    layer = RNOBlock(
        n_modes=modes,
        hidden_channels=width,
        norm=norm,
        complex_data=complex_data,
    )

    dtype = torch.cfloat if complex_data else torch.float32
    x = torch.randn(batch_size, num_timesteps, width, *size, dtype=dtype)

    out = layer(x)
    assert out.shape == (batch_size, width, *size)
    assert out.dtype == dtype


@pytest.mark.parametrize("stabilizer", [None, "tanh"])
@pytest.mark.parametrize("complex_data", [False, True])
def test_RNOBlock_stabilizer(stabilizer, complex_data):
    """Test RNOBlock with stabilizer options"""
    modes = (8, 8)
    width = 16
    size = [12, 12]
    batch_size = 2
    num_timesteps = 3

    layer = RNOBlock(
        n_modes=modes,
        hidden_channels=width,
        stabilizer=stabilizer,
        complex_data=complex_data,
    )

    dtype = torch.cfloat if complex_data else torch.float32
    x = torch.randn(batch_size, num_timesteps, width, *size, dtype=dtype)

    out = layer(x)
    assert out.shape == (batch_size, width, *size)
    assert out.dtype == dtype


@pytest.mark.parametrize("separable", [False, True])
@pytest.mark.parametrize("complex_data", [False, True])
def test_RNOBlock_separable(separable, complex_data):
    """Test RNOBlock with separable convolutions"""
    modes = (8, 8)
    width = 16
    size = [12, 12]
    batch_size = 2
    num_timesteps = 3

    layer = RNOBlock(
        n_modes=modes,
        hidden_channels=width,
        separable=separable,
        complex_data=complex_data,
    )

    dtype = torch.cfloat if complex_data else torch.float32
    x = torch.randn(batch_size, num_timesteps, width, *size, dtype=dtype)

    out = layer(x)
    assert out.shape == (batch_size, width, *size)
    assert out.dtype == dtype


@pytest.mark.parametrize("preactivation", [False, True])
@pytest.mark.parametrize("complex_data", [False, True])
def test_RNOBlock_preactivation(preactivation, complex_data):
    """Test RNOBlock with preactivation"""
    modes = (8, 8)
    width = 16
    size = [12, 12]
    batch_size = 2
    num_timesteps = 3

    layer = RNOBlock(
        n_modes=modes,
        hidden_channels=width,
        preactivation=preactivation,
        complex_data=complex_data,
    )

    dtype = torch.cfloat if complex_data else torch.float32
    x = torch.randn(batch_size, num_timesteps, width, *size, dtype=dtype)

    out = layer(x)
    assert out.shape == (batch_size, width, *size)
    assert out.dtype == dtype


@pytest.mark.parametrize("factorization", [None, "Tucker"])
@pytest.mark.parametrize("complex_data", [False, True])
def test_RNOBlock_factorization(factorization, complex_data):
    """Test RNOBlock with factorization"""
    modes = (8, 8)
    width = 16
    size = [12, 12]
    batch_size = 2
    num_timesteps = 3

    layer = RNOBlock(
        n_modes=modes,
        hidden_channels=width,
        factorization=factorization,
        rank=0.5 if factorization else 1.0,
        complex_data=complex_data,
    )

    dtype = torch.cfloat if complex_data else torch.float32
    x = torch.randn(batch_size, num_timesteps, width, *size, dtype=dtype)

    out = layer(x)
    assert out.shape == (batch_size, width, *size)
    assert out.dtype == dtype
