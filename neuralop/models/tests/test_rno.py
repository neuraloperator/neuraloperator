import torch
from neuralop.models import RNO
import pytest
from math import prod

@pytest.mark.parametrize('n_dim', [1, 2, 3])
@pytest.mark.parametrize('lifting_channel_ratio', [1, 2])
@pytest.mark.parametrize('projection_channel_ratio', [1, 2])
def test_rno(n_dim, lifting_channel_ratio, projection_channel_ratio):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    s = 32
    modes = 8
    width = 16
    n_layers = 3
    residual = False

    batch_size = 2
    size = (s, )*n_dim
    num_time_steps = 5
    n_modes = (modes,)*n_dim
    model = RNO(n_modes,
                in_channels=3,
                out_channels=1,
                hidden_channels=width,
                n_layers=n_layers,
                lifting_channel_ratio=lifting_channel_ratio,
                projection_channel_ratio=projection_channel_ratio,
                residual=residual,
                domain_padding=None,
                separable=False,
                factorization=None
                ).to(device)
    in_data = torch.randn(batch_size, num_time_steps, 3, *size).to(device)

    # Test forward pass
    out, _ = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]

    # Check backward pass
    loss = out.sum()
    loss.backward()

    n_unused_params = 0
    for param in model.parameters():
        if param.grad is None:
            n_unused_params += 1
    assert n_unused_params == 0, f'{n_unused_params} parameters were unused!'

@pytest.mark.parametrize('resolution_scaling_factor', 
                         [[2, 1, 1], [1, 2, 1], [1, 1, 2], [1, 2, 2], [1, 0.5, 1]])
def test_rno_superresolution(resolution_scaling_factor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = 16
    modes = 16
    width = 15
    n_layers = 3
    residual = False
    
    batch_size = 3
    n_dim = 2
    size = (s, )*n_dim
    num_time_steps = 5
    n_modes = (modes,)*n_dim

    model = RNO(n_modes,
                in_channels=3,
                out_channels=1,
                hidden_channels=width,
                n_layers=n_layers,
                residual=residual,
                domain_padding=None,
                resolution_scaling_factor=resolution_scaling_factor,
                separable=False,
                factorization=None
                ).to(device)

    in_data = torch.randn(batch_size, num_time_steps, 3, *size).to(device)
    # Test forward pass
    out, _ = model(in_data)

    # Check output size
    factor = prod(resolution_scaling_factor)
    
    assert list(out.shape) == [batch_size, 1] + [int(round(factor*s)) for s in size]