import pytest
import torch
import torch.nn as nn

from ..mwno import MWNO


@pytest.mark.parametrize("n_dim", [1, 2])
@pytest.mark.parametrize("alpha", [3, 5, 8])
@pytest.mark.parametrize("k", [3, 4])
@pytest.mark.parametrize("c", [1, 2])
@pytest.mark.parametrize("n_layers", [2, 3])
@pytest.mark.parametrize("L", [0, 1])
@pytest.mark.parametrize("base", ["legendre"])
@pytest.mark.parametrize("lifting_channels", [0, 128])
@pytest.mark.parametrize("projection_channels", [0, 128])
def test_mwno(
    n_dim,
    alpha,
    k,
    c,
    n_layers,
    L,
    base,
    lifting_channels,
    projection_channels
):
    
    if torch.cuda.is_available():
        device = "cuda"
        s = 16
        batch_size = 4
    else:
        device = "cpu"
        s = 16
        batch_size = 3
    
    # Ensure size is power of 2 for wavelet transform
    s = 2 ** int(torch.log2(torch.tensor(s, dtype=torch.float)).item())
    
    # Create model based on dimension
    if n_dim == 1:
        n_modes = alpha
    elif n_dim == 2:
        n_modes = (alpha, alpha)
    elif n_dim == 3:
        n_modes = (alpha, alpha, alpha)
    
    model = MWNO(
        n_modes=n_modes,
        in_channels=3,
        out_channels=1,
        k=k,
        c=c,
        n_layers=n_layers,
        L=L,
        lifting_channels=lifting_channels,
        projection_channels=projection_channels,
        base=base,
    ).to(device)
    
    # Create input data in MWNO's expected format: (B, *spatial, channels)
    if n_dim == 1:
        in_data = torch.randn(batch_size, s, 3).to(device)
        expected_shape = [batch_size, s]
    elif n_dim == 2:
        in_data = torch.randn(batch_size, s, s, 3).to(device)
        expected_shape = [batch_size, s, s]
    elif n_dim == 3:
        in_data = torch.randn(batch_size, s, s, s, 3).to(device)
        expected_shape = [batch_size, s, s, s]
    
    # Test forward pass
    out = model(in_data)
    
    # Check output size
    assert list(out.shape) == expected_shape, f"Expected shape {expected_shape}, got {list(out.shape)}"
    
    # Check backward pass
    loss = out.sum()
    loss.backward()
    
    # Check that all parameters have gradients
    n_unused_params = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            n_unused_params += 1
            print(f"Unused parameter: {name}")
    
    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"