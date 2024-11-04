from math import prod

import pytest
import torch
from tensorly import tenalg
from configmypy import Bunch

from neuralop.models import LocalFNO

tenalg.set_backend("einsum")

@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("mix_derivatives", [True, False])
@pytest.mark.parametrize("fin_diff_implementation", ["subtract_middle", "subtract_all"])
def test_local_fno_without_disco(
    n_dim,
    mix_derivatives,
    fin_diff_implementation
):
    if torch.has_cuda:
        device = "cuda"
        s = 32
        modes = 8
        width = 16
        fc_channels = 16
        batch_size = 4
        n_layers = 4
    else:
        device = "cpu"
        fno_block_precision = "full"
        s = 32
        modes = 5
        width = 15
        fc_channels = 32
        batch_size = 3
        n_layers = 2

    rank = 0.2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    conv_padding_mode = 'zeros'
    model = LocalFNO(
        in_channels=3,
        out_channels=1,
        default_in_shape=size,
        diff_layers=True,
        mix_derivatives=mix_derivatives,
        fin_diff_implementation=fin_diff_implementation,
        disco_layers=False,
        conv_padding_mode=conv_padding_mode,
        hidden_channels=width,
        n_modes=n_modes,
        rank=rank,
        fixed_rank_modes=False,
        n_layers=n_layers,
        fc_channels=fc_channels,
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

@pytest.mark.parametrize('disco_layers', [True,
                                          [True, False, True, False],
                                          [False, False, False, True]])
def test_local_fno_with_disco(
    disco_layers
):
    
    n_dim = 2
    if torch.has_cuda:
        device = "cuda"
        s = 32
        modes = 8
        width = 16
        fc_channels = 16
        batch_size = 4
        n_layers = 4
    else:
        device = "cpu"
        s = 32
        modes = 5
        width = 15
        fc_channels = 32
        batch_size = 3
        n_layers = 2

    rank = 0.2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    
    model = LocalFNO(
        in_channels=3,
        out_channels=1,
        default_in_shape=size,
        disco_layers=disco_layers,
        hidden_channels=width,
        n_modes=n_modes,
        rank=rank,
        fixed_rank_modes=False,
        n_layers=n_layers,
        fc_channels=fc_channels,
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