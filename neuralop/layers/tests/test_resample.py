import math
import torch
from ..resample import resample, iterative_resample


def test_odd_grid_multidimensional_trig():
    """Verify resample on odd grids against an exact analytical trigonometric mode."""
    N = 5
    M = 9
    dtype = torch.float64

    grid_1d = 2.0 * math.pi * torch.arange(N, dtype=dtype) / N
    gx, gy, gz = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing="ij")
    x = (
        (torch.cos(2.0 * gx) * torch.cos(gy) * torch.sin(2.0 * gz))
        .unsqueeze(0)
        .unsqueeze(0)
    )

    res_scale = [M / N] * 3
    axis = [-3, -2, -1]
    out = resample(x, res_scale, axis)

    new_grid_1d = 2.0 * math.pi * torch.arange(M, dtype=dtype) / M
    ngx, ngy, ngz = torch.meshgrid(new_grid_1d, new_grid_1d, new_grid_1d, indexing="ij")
    expected = (
        (torch.cos(2.0 * ngx) * torch.cos(ngy) * torch.sin(2.0 * ngz))
        .unsqueeze(0)
        .unsqueeze(0)
    )

    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-10)


def test_nyquist_even_up_down_amplitude():
    """Verify Nyquist amplitude handling during upsampling and downsampling on even grids."""
    dtype = torch.float64

    # 1. Upsampling from even N=8 to M=16
    N = 8
    M = 16
    n = torch.arange(N, dtype=dtype)
    x = torch.cos(math.pi * n).reshape(1, 1, N)

    out_up = iterative_resample(x, res_scale=2, axis=-1)
    m = torch.arange(M, dtype=dtype)
    expected_up = torch.cos(math.pi * 0.5 * m).reshape(1, 1, M)

    assert out_up.shape == (1, 1, M)
    assert torch.allclose(out_up, expected_up, atol=1e-10)
    assert math.isclose(out_up.abs().max().item(), 1.0, rel_tol=1e-7)

    # 2. Downsampling from even M=16 to N=8
    out_down = iterative_resample(out_up, res_scale=0.5, axis=-1)
    expected_down = x

    assert out_down.shape == (1, 1, N)
    assert torch.allclose(out_down, expected_down, atol=1e-10)
    assert math.isclose(out_down.abs().max().item(), 1.0, rel_tol=1e-7)


def test_float64_precision_and_gradcheck():
    """Verify float64 precision retention and autograd gradient correctness via gradcheck."""
    dtype = torch.float64
    x = (torch.arange(64, dtype=dtype).reshape(1, 1, 4, 4, 4) / 64).requires_grad_()

    out = resample(x, [1.5, 1.5, 1.5], [-3, -2, -1])
    assert out.dtype == torch.float64

    def func(inp):
        return resample(inp, [1.5, 1.5, 1.5], [-3, -2, -1])

    assert torch.autograd.gradcheck(func, x, eps=1e-6, atol=1e-5)

    loss = (out * torch.ones_like(out)).sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.dtype == torch.float64


def test_iterative_listaxis_usable_and_equivalent():
    """Verify iterative_resample accepts list axis on ND tensors and matches analytical output."""
    N = 5
    M = 7
    dtype = torch.float64

    grid_1d = 2.0 * math.pi * torch.arange(N, dtype=dtype) / N
    gx, gy, gz = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing="ij")
    x = (torch.cos(gx) * torch.sin(gy) * torch.cos(2.0 * gz)).unsqueeze(0).unsqueeze(0)

    axis = [-3, -2, -1]
    res_scale = [M / N] * 3
    out_iter = iterative_resample(x, res_scale, axis)
    out_resample = resample(x, res_scale, axis)

    new_grid_1d = 2.0 * math.pi * torch.arange(M, dtype=dtype) / M
    ngx, ngy, ngz = torch.meshgrid(new_grid_1d, new_grid_1d, new_grid_1d, indexing="ij")
    expected = (
        (torch.cos(ngx) * torch.sin(ngy) * torch.cos(2.0 * ngz))
        .unsqueeze(0)
        .unsqueeze(0)
    )

    assert torch.allclose(out_iter, expected, atol=1e-10)
    assert torch.allclose(out_iter, out_resample, atol=1e-10)
