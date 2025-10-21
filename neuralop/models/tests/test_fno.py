from math import prod

import pytest
import torch
import torch.nn.functional as F
from tensorly import tenalg

from neuralop.models import FNO

tenalg.set_backend("einsum")


@pytest.mark.parametrize("n_dim", [1, 2, 3, 4])
@pytest.mark.parametrize("fno_block_precision", ["full"])
@pytest.mark.parametrize("stabilizer", [None, "tanh"])
@pytest.mark.parametrize("lifting_channel_ratio", [1, 2])
@pytest.mark.parametrize("preactivation", [False, True])
@pytest.mark.parametrize("complex_data", [True, False])
def test_fno(
    n_dim,
    fno_block_precision,
    stabilizer,
    lifting_channel_ratio,
    preactivation,
    complex_data,
):
    if torch.has_cuda:
        device = "cuda"
        s = 16
        modes = 8
        width = 16
        batch_size = 4
        n_layers = 4
    else:
        device = "cpu"
        fno_block_precision = "full"
        s = 16
        modes = 5
        width = 15
        batch_size = 3
        n_layers = 2

    dtype = torch.cfloat if complex_data else torch.float32
    rank = 0.2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    model = FNO(
        in_channels=3,
        out_channels=1,
        hidden_channels=width,
        n_modes=n_modes,
        rank=rank,
        fixed_rank_modes=False,
        n_layers=n_layers,
        stabilizer=stabilizer,
        lifting_channel_ratio=lifting_channel_ratio,
        preactivation=preactivation,
        complex_data=complex_data,
        fno_block_precision=fno_block_precision,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size, dtype=dtype).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]

    # Check backward pass
    loss = out.sum()
    # take the modulus if data is complex-valued to create grad
    if dtype == torch.cfloat:
        loss = (loss.real**2 + loss.imag**2) ** 0.5
    loss.backward()

    n_unused_params = 0
    for param in model.parameters():
        if param.grad is None:
            n_unused_params += 1
    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"


@pytest.mark.parametrize(
    "resolution_scaling_factor",
    [
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2],
        [1, 2, 2],
        [1, 0.5, 1],
    ],
)
def test_fno_superresolution(resolution_scaling_factor):
    device = "cpu"
    s = 16
    modes = 5
    hidden_channels = 15
    batch_size = 3
    n_layers = 3
    n_dim = 2
    rank = 0.2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim

    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        factorization="cp",
        implementation="reconstructed",
        rank=rank,
        resolution_scaling_factor=resolution_scaling_factor,
        n_layers=n_layers,
    ).to(device)

    print(f"{model.resolution_scaling_factor=}")

    in_data = torch.randn(batch_size, 3, *size).to(device)
    # Test forward pass
    out = model(in_data)

    # Check output size
    factor = prod(resolution_scaling_factor)

    assert list(out.shape) == [batch_size, 1] + [int(round(factor * s)) for s in size]


@pytest.mark.parametrize("norm", [None, "group_norm", "instance_norm"])
@pytest.mark.parametrize("use_channel_mlp", [True, False])
@pytest.mark.parametrize("channel_mlp_skip", ["linear", "identity", "soft-gating", None])
@pytest.mark.parametrize("fno_skip", ["linear", "identity", "soft-gating", None])
@pytest.mark.parametrize("complex_data", [True, False])
def test_fno_advanced_params(norm, use_channel_mlp, channel_mlp_skip, fno_skip, complex_data):
    """Test FNO with various advanced parameter combinations."""
    device = "cpu"
    s = 16
    modes = 5
    hidden_channels = 15
    batch_size = 3
    n_layers = 2
    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim

    dtype = torch.cfloat if complex_data else torch.float32

    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        norm=norm,
        use_channel_mlp=use_channel_mlp,
        channel_mlp_skip=channel_mlp_skip,
        fno_skip=fno_skip,
        complex_data=complex_data,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size, dtype=dtype).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]

    # Test backward pass
    loss = out.sum()
    # take the modulus if data is complex-valued to create grad
    if dtype == torch.cfloat:
        loss = (loss.real**2 + loss.imag**2) ** 0.5
    loss.backward()


@pytest.mark.parametrize("positional_embedding", ["grid", None])
@pytest.mark.parametrize("domain_padding", [None, 0.1, [0.1, 0.2]])
def test_fno_embedding_and_padding(positional_embedding, domain_padding):
    """Test FNO with different positional embeddings and domain padding."""
    device = "cpu"
    s = 16
    modes = 5
    hidden_channels = 15
    projection_channel_ratio = 2
    lifting_channel_ratio = 2
    batch_size = 3
    n_layers = 2
    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim

    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        positional_embedding=positional_embedding,
        domain_padding=domain_padding,
        projection_channel_ratio=projection_channel_ratio,
        lifting_channel_ratio=lifting_channel_ratio,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]


@pytest.mark.parametrize("channel_mlp_dropout", [0.0, 0.1, 0.5])
@pytest.mark.parametrize("channel_mlp_expansion", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("non_linearity", [F.gelu, F.relu, F.tanh])
def test_fno_channel_mlp_params(channel_mlp_dropout, channel_mlp_expansion, non_linearity):
    """Test FNO with different channel MLP parameters."""
    device = "cpu"
    s = 16
    modes = 5
    hidden_channels = 15
    batch_size = 3
    n_layers = 2
    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim

    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        channel_mlp_dropout=channel_mlp_dropout,
        channel_mlp_expansion=channel_mlp_expansion,
        non_linearity=non_linearity,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]
