from math import prod

import pytest
import torch
from tensorly import tenalg

try:
    import torch_harmonics
except ModuleNotFoundError:
    pytest.skip("Skipping because torch_harmonics is not installed", allow_module_level=True)

from neuralop.models import LocalNO

tenalg.set_backend("einsum")


@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("mix_derivatives", [True, False])
def test_local_no_without_disco(
    n_dim,
    mix_derivatives,
):
    if torch.cuda.is_available():
        device = "cuda"
        s = 32
        modes = 8
        width = 16
        batch_size = 4
        n_layers = 4
    else:
        device = "cpu"
        fno_block_precision = "full"
        s = 32
        modes = 5
        width = 15
        batch_size = 3
        n_layers = 2

    rank = 0.2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    conv_padding_mode = "zeros"
    model = LocalNO(
        in_channels=3,
        out_channels=1,
        default_in_shape=size,
        diff_layers=True,
        mix_derivatives=mix_derivatives,
        disco_layers=False,
        conv_padding_mode=conv_padding_mode,
        hidden_channels=width,
        n_modes=n_modes,
        rank=rank,
        fixed_rank_modes=False,
        n_layers=n_layers,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size).to(device)

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

@pytest.mark.parametrize('disco_layers', [
    True,
    [True, False, True, False],
    [False, False, False, True]
])
@pytest.mark.parametrize('disco_kernel_shape', [
    [2,4],
    [3,3],
])
def test_local_no_with_disco(
    disco_layers,
    disco_kernel_shape
):
    
    n_dim = 2
    if torch.cuda.is_available():
        device = "cuda"
        s = 32
        modes = 8
        width = 16
        batch_size = 4
        n_layers = 4
    else:
        device = "cpu"
        s = 32
        modes = 5
        width = 15
        batch_size = 3
        n_layers = 2

    if isinstance(disco_layers, list):
        disco_layers = disco_layers[:n_layers]
    rank = 0.2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim

    model = LocalNO(
        in_channels=3,
        out_channels=1,
        default_in_shape=size,
        disco_layers=disco_layers,
        disco_kernel_shape=disco_kernel_shape,
        hidden_channels=width,
        n_modes=n_modes,
        rank=rank,
        fixed_rank_modes=False,
        n_layers=n_layers,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size).to(device)

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


def test_local_no_group_norm():
    """Test LocalNO with group_norm and custom norm_groups"""
    norm_groups = 4
    hidden_channels = 16
    size = (12, 12)
    model = LocalNO(
        in_channels=3,
        out_channels=1,
        default_in_shape=size,
        hidden_channels=hidden_channels,
        n_modes=(5, 5),
        n_layers=2,
        diff_layers=True,
        disco_layers=False,
        conv_padding_mode="zeros",
        norm="group_norm",
        norm_groups=norm_groups,
    )

    for norm_layer in model.local_no_blocks.norm:
        assert isinstance(norm_layer, torch.nn.GroupNorm)
        assert norm_layer.num_groups == norm_groups
        assert norm_layer.num_channels == hidden_channels
