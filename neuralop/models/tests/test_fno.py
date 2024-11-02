from math import prod

import pytest
import torch
from tensorly import tenalg
from configmypy import Bunch

from neuralop import TFNO
from neuralop.models import FNO

tenalg.set_backend("einsum")


@pytest.mark.parametrize(
    "factorization", ["ComplexDense", "ComplexTucker", "ComplexCP", "ComplexTT"]
)
@pytest.mark.parametrize("implementation", ["factorized", "reconstructed"])
@pytest.mark.parametrize("n_dim", [1, 2, 3, 4])
@pytest.mark.parametrize("fno_block_precision", ["full", "half", "mixed"])
@pytest.mark.parametrize("stabilizer", [None, "tanh"])
@pytest.mark.parametrize("lifting_channel_ratio", [1, 2])
@pytest.mark.parametrize("preactivation", [False, True])
@pytest.mark.parametrize("complex_data", [True, False])
def test_tfno(
    factorization,
    implementation,
    n_dim,
    fno_block_precision,
    stabilizer,
    lifting_channel_ratio,
    preactivation,
    complex_data
):
    if torch.has_cuda:
        device = "cuda"
        s = 16
        modes = 8
        width = 16
        fc_channels = 16
        batch_size = 4
        n_layers = 4
    else:
        device = "cpu"
        fno_block_precision = "full"
        s = 16
        modes = 5
        width = 15
        fc_channels = 32
        batch_size = 3
        n_layers = 2

    dtype = torch.cfloat if complex_data else torch.float32
    rank = 0.2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    model = TFNO(
        in_channels=3,
        out_channels=1,
        hidden_channels=width,
        n_modes=n_modes,
        factorization=factorization,
        implementation=implementation,
        rank=rank,
        fixed_rank_modes=False,
        n_layers=n_layers,
        stabilizer=stabilizer,
        fc_channels=fc_channels,
        lifting_channel_ratio=lifting_channel_ratio,
        preactivation=preactivation,
        complex_data=complex_data,
        fno_block_precision=fno_block_precision
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
        loss = (loss.real ** 2 + loss.imag ** 2) ** 0.5
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
    fc_channels = 32
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
        fc_channels=fc_channels,
    ).to(device)

    print(f"{model.resolution_scaling_factor=}")

    in_data = torch.randn(batch_size, 3, *size).to(device)
    # Test forward pass
    out = model(in_data)

    # Check output size
    factor = prod(resolution_scaling_factor)

    assert list(out.shape) == [batch_size, 1] + [int(round(factor * s)) for s in size]
