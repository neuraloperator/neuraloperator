import pytest
import torch
from tltorch import FactorizedTensor
from ..spherical_convolution import SphericalConv
from ..spherical_convolution import SHT

@pytest.mark.parametrize('factorization', ['ComplexDense', 'ComplexCP', 'ComplexTucker', 'ComplexTT'])
@pytest.mark.parametrize('implementation', ['factorized', 'reconstructed'])
def test_SphericalConv(factorization, implementation):
    """Test for SphericalConv (2D only)
    
    Compares Factorized and Dense convolution output
    Verifies that a dense conv and factorized conv with the same weight produce the same output

    Checks the output size

    Verifies that dynamically changing the number of Fourier modes doesn't break the conv
    """
    n_modes = (6, 6)

    conv = SphericalConv(
        3, 3, n_modes, n_layers=1, bias=False, implementation=implementation, factorization=factorization)

    conv_dense = SphericalConv(
        3, 3, n_modes, n_layers=1, bias=False, implementation='reconstructed', factorization=None)

    conv_dense.weight[0] = FactorizedTensor.from_tensor(conv.weight[0].to_tensor(), rank=None, factorization='ComplexDense')
    x = torch.randn(2, 3, *(12, 12))

    res_dense = conv_dense(x)
    res = conv(x)

    torch.testing.assert_close(res_dense, res)

    # Downsample outputs
    block = SphericalConv(
        3, 4, n_modes, n_layers=1, output_scaling_factor=0.5)

    x = torch.randn(2, 3, *(12, 12))
    res = block(x)
    assert(list(res.shape[2:]) == [12//2, 12//2])

    # Upsample outputs
    block = SphericalConv(
        3, 4, n_modes, n_layers=1, output_scaling_factor=2)

    x = torch.randn(2, 3, *(12, 12))
    res = block(x)
    assert res.shape[1] == 4 # Check out channels
    assert(list(res.shape[2:]) == [12*2, 12*2])

    # Test change of grid
    block = SphericalConv(
        4, 4, n_modes, n_layers=2, sht_grids=["equiangular", "legendre-gauss", "equiangular"])
    x = torch.randn(2, 4, *(12, 12))
    res = block[0](x)
    res = block[1](res)
    assert(res.shape[2:] == x.shape[2:])

    res = block[0].transform(x)
    res = block[1].transform(res)
    assert(res.shape[2:] == x.shape[2:])


@pytest.mark.parametrize('grid', ['equiangular', 'legendre-gauss'])
def test_sht(grid):
    nlat = 16
    nlon = 2*nlat
    batch_size = 2
    if grid == "equiangular":
        mmax = nlat // 2
    else:
        mmax = nlat
    lmax = mmax
    norm = 'ortho'
    dtype = torch.float32

    sht_handle = SHT(dtype=dtype)

    # Create input
    coeffs = torch.zeros(batch_size, lmax, mmax, dtype=torch.complex64)
    coeffs[:, :lmax, :mmax] = torch.randn(batch_size, lmax, mmax, dtype=torch.complex64)
    
    signal = sht_handle.isht(coeffs, s=(nlat, nlon), grid=grid, norm=norm).to(torch.float32)

    coeffs = sht_handle.sht(signal, s=(lmax, mmax), grid=grid, norm=norm)
    rec = sht_handle.isht(coeffs, s=(nlat, nlon), grid=grid, norm=norm)
    torch.testing.assert_close(signal, rec, rtol=1e-4, atol=1e-4)
