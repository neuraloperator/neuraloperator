import torch
import pytest

from ..otno import OTNO

# Fixed variables
in_channels = 9
out_channels = 1
hidden_channels = 64
n_modes = [16, 16]
lifting_channel_ratio = 3
projection_channel_ratio = 3
use_channel_mlp = True
norm = "group_norm"
factorization = "tucker"
n_layers = 4
domain_padding = 0.2

# data parameters
n_s_sqrt = 18
n_target_points = 100


@pytest.mark.parametrize("positional_embedding", [None, "grid"])
def test_otno(positional_embedding):
    if torch.backends.cuda.is_built():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu:0")

    # OTNO is designed for batch_size = 1
    batch_size = 1

    model = OTNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        lifting_channel_ratio=lifting_channel_ratio,
        projection_channel_ratio=projection_channel_ratio,
        # Parameters from OTNO + FNO
        use_channel_mlp=use_channel_mlp,
        channel_mlp_expansion=1.0 if use_channel_mlp else None,
        channel_mlp_dropout=0,
        norm=norm,
        factorization=factorization,
        n_layers=n_layers,
        positional_embedding=positional_embedding,
        domain_padding=domain_padding,
    ).to(device)

    # Create input data
    x_shape = [batch_size, in_channels, n_s_sqrt, n_s_sqrt]
    x = torch.randn(*x_shape, device=device)
    x.requires_grad_(True)

    # Create decoding indices
    n_latent_points = n_s_sqrt * n_s_sqrt
    ind_dec = torch.randint(0, n_latent_points, (n_target_points,), device=device)

    # Test forward pass
    out = model(x=x, ind_dec=ind_dec)

    # Check output size (batch_size fixed to 1)
    assert list(out.shape) == [batch_size, n_target_points]

    # Check backward pass
    assert out.isfinite().all()
    loss = out.sum()
    loss.backward()
    n_unused_params = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)
            n_unused_params += 1
    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"
    # No multi-batch gradient checks needed since batch_size == 1
