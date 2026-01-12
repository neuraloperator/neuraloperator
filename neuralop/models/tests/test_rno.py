import torch
import torch.nn.functional as F
from neuralop.models import RNO
import pytest
from math import prod


@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("fno_block_precision", ["full"])
@pytest.mark.parametrize("stabilizer", [None, "tanh"])
@pytest.mark.parametrize("lifting_channel_ratio", [1, 2])
@pytest.mark.parametrize("preactivation", [False, True])
def test_rno(
    n_dim,
    fno_block_precision,
    stabilizer,
    lifting_channel_ratio,
    preactivation,
):
    if torch.cuda.is_available():
        device = "cuda"
        s = 16
        modes = 8
        width = 16
        batch_size = 4
        n_layers = 4
    else:
        device = "cpu"
        fno_block_precision = "full"
        s = 12
        modes = 5
        width = 15
        batch_size = 2
        n_layers = 2

    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    num_time_steps = 3
    model = RNO(
        in_channels=3,
        out_channels=1,
        hidden_channels=width,
        n_modes=n_modes,
        n_layers=n_layers,
        stabilizer=stabilizer,
        lifting_channel_ratio=lifting_channel_ratio,
        preactivation=preactivation,
        fno_block_precision=fno_block_precision,
    ).to(device)

    in_data = torch.randn(batch_size, num_time_steps, 3, *size).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]

    # Check backward pass
    loss = out.sum()
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
def test_rno_superresolution(resolution_scaling_factor):
    device = "cpu"
    s = 12
    modes = 5
    hidden_channels = 15
    batch_size = 2
    n_layers = 3
    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    num_time_steps = 3

    model = RNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        resolution_scaling_factor=resolution_scaling_factor,
        n_layers=n_layers,
    ).to(device)

    print(f"{model.resolution_scaling_factor=}")

    in_data = torch.randn(batch_size, num_time_steps, 3, *size).to(device)
    # Test forward pass
    out = model(in_data)

    # Check output size
    factor = prod(resolution_scaling_factor)

    assert list(out.shape) == [batch_size, 1] + [int(round(factor * s)) for s in size]


@pytest.mark.parametrize("norm", [None, "group_norm", "instance_norm"])
@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("channel_mlp_skip", ["linear", "soft-gating", None])
def test_rno_channel_mlp_params_advanced(norm, complex_data, channel_mlp_skip):
    """Test RNO with channel MLP and various advanced parameter combinations."""
    device = "cpu"
    s = 12
    modes = 5
    hidden_channels = 15
    batch_size = 2
    n_layers = 2
    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    num_time_steps = 3

    dtype = torch.cfloat if complex_data else torch.float32

    model = RNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        norm=norm,
        complex_data=complex_data,
        use_channel_mlp=True,
        channel_mlp_skip=channel_mlp_skip,
    ).to(device)

    in_data = torch.randn(batch_size, num_time_steps, 3, *size, dtype=dtype).to(device)

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


@pytest.mark.parametrize("norm", [None, "group_norm"])
@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("fno_skip", ["linear", "soft-gating", None])
def test_rno_fno_skip_params(norm, complex_data, fno_skip):
    """Test RNO with FNO skip connections and various parameter combinations."""
    device = "cpu"
    s = 12
    modes = 5
    hidden_channels = 15
    batch_size = 2
    n_layers = 2
    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    num_time_steps = 3

    dtype = torch.cfloat if complex_data else torch.float32

    model = RNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        norm=norm,
        complex_data=complex_data,
        use_channel_mlp=False,
        fno_skip=fno_skip,
    ).to(device)

    in_data = torch.randn(batch_size, num_time_steps, 3, *size, dtype=dtype).to(device)

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
def test_rno_embedding_and_padding(positional_embedding, domain_padding):
    """Test RNO with different positional embeddings and domain padding."""
    device = "cpu"
    s = 12
    modes = 5
    hidden_channels = 15
    projection_channel_ratio = 2
    lifting_channel_ratio = 2
    batch_size = 2
    n_layers = 2
    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    num_time_steps = 3

    model = RNO(
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

    in_data = torch.randn(batch_size, num_time_steps, 3, *size).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]


@pytest.mark.parametrize("channel_mlp_dropout", [0.0, 0.1])
@pytest.mark.parametrize("channel_mlp_expansion", [0.5, 1.0])
@pytest.mark.parametrize("non_linearity", [F.gelu, F.relu, F.tanh])
def test_rno_channel_mlp_params(channel_mlp_dropout, channel_mlp_expansion, non_linearity):
    """Test RNO with different channel MLP parameters."""
    device = "cpu"
    s = 12
    modes = 5
    hidden_channels = 15
    batch_size = 2
    n_layers = 2
    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    num_time_steps = 3

    model = RNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        channel_mlp_dropout=channel_mlp_dropout,
        channel_mlp_expansion=channel_mlp_expansion,
        non_linearity=non_linearity,
    ).to(device)

    in_data = torch.randn(batch_size, num_time_steps, 3, *size).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]


def test_rno_predict():
    """Test RNO prediction functionality."""
    device = "cpu"
    s = 12
    modes = 5
    hidden_channels = 15
    batch_size = 2
    n_layers = 2
    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    num_time_steps = 3
    num_steps = 1

    model = RNO(
        in_channels=3,
        out_channels=3,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
    ).to(device)

    in_data = torch.randn(batch_size, num_time_steps, 3, *size).to(device)

    # Test predict method
    predictions = model.predict(in_data, num_steps)

    # Check output size
    expected_shape = [batch_size, num_steps, 3] + list(size)
    assert list(predictions.shape) == expected_shape
