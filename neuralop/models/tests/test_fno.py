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
@pytest.mark.parametrize("lifting_channels", [None, 32])
@pytest.mark.parametrize("preactivation", [False, True])
@pytest.mark.parametrize("complex_data", [False, True])
def test_tfno(
    factorization,
    implementation,
    n_dim,
    fno_block_precision,
    stabilizer,
    lifting_channels,
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
        use_channel_mlp = True
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

        use_channel_mlp = True

    rank = 0.2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    model = TFNO(
        hidden_channels=width,
        n_modes=n_modes,
        factorization=factorization,
        implementation=implementation,
        rank=rank,
        fixed_rank_modes=False,
        joint_factorization=False,
        n_layers=n_layers,
        fno_block_precision=fno_block_precision,
        use_channel_mlp=use_channel_mlp,
        stabilizer=stabilizer,
        fc_channels=fc_channels,
        lifting_channels=lifting_channels,
        preactivation=preactivation,
        complex_data=complex_data
    ).to(device)

    if complex_data:
        in_data = torch.randn(batch_size, 3, *size, dtype=torch.cfloat).to(device)
    else:
        in_data = torch.randn(batch_size, 3, *size).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]

    # Check backward pass
    loss = out.sum()
    # take the modulus if data is complex-valued to create grad
    if complex_data:
        loss = (loss.real ** 2 + loss.imag ** 2) ** 0.5
    loss.backward()

    n_unused_params = 0
    for param in model.parameters():
        if param.grad is None:
            n_unused_params += 1
    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"


@pytest.mark.parametrize(
    "output_scaling_factor",
    [
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2],
        [1, 2, 2],
        [1, 0.5, 1],
    ],
)
def test_fno_superresolution(output_scaling_factor):
    device = "cpu"
    s = 16
    modes = 5
    hidden_channels = 15
    fc_channels = 32
    batch_size = 3
    n_layers = 3
    use_channel_mlp = True
    n_dim = 2
    rank = 0.2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim

    model = FNO(
        n_modes,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        factorization="cp",
        implementation="reconstructed",
        rank=rank,
        output_scaling_factor=output_scaling_factor,
        n_layers=n_layers,
        use_channel_mlp=use_channel_mlp,
        fc_channels=fc_channels,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size).to(device)
    # Test forward pass
    out = model(in_data)

    # Check output size
    factor = prod(output_scaling_factor)

    assert list(out.shape) == [batch_size, 1] + [int(round(factor * s)) for s in size]


def test_max_n_modes():
    device = "cpu"
    modes = [5, 5, 5, 5]
    dim = [1, 2, 3, 4]
    max_n_modes=[6, 6, 6, 6]
    max_n_modes_updated=[8, 8, 8, 8]
    hidden_channels = 15
    fc_channels = 32
    n_layers = 3
    use_channel_mlp = True
    n_dim = 2
    rank = 0.2

    for i in dim:
        model = FNO(
            n_modes=modes[:i],
            hidden_channels=hidden_channels,
            max_n_modes=max_n_modes[:i],
            in_channels=3,
            out_channels=1,
            factorization="cp",
            implementation="reconstructed",
            rank=rank,
            n_layers=n_layers,
            use_channel_mlp=use_channel_mlp,
            fc_channels=fc_channels,
        ).to(device)

        assert model.max_n_modes == max_n_modes[:i] # check original max setting
        
        model.max_n_modes = max_n_modes_updated[:i]
        
        assert model.max_n_modes == max_n_modes_updated[:i] # check new max setting
        