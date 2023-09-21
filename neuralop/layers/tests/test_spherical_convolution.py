import pytest
import torch
from tltorch import FactorizedTensor
from ..spherical_convolution import SphericalConv


@pytest.mark.parametrize('factorization', ['ComplexDense', 'ComplexCP', 'ComplexTucker', 'ComplexTT'])
@pytest.mark.parametrize('implementation', ['factorized', 'reconstructed'])
def test_SphericalConv(factorization, implementation):
    """Test for SphericalConv (2D only)
    
    Compares Factorized and Dense convolution output
    Verifies that a dense conv and factorized conv with the same weight produce the same output

    Checks the output size

    Verifies that dynamically changing the number of Fourier modes doesn't break the conv
    """
    n_modes = (10, 8)

    conv = SphericalConv(
        3, 3, n_modes, n_layers=1, bias=False, implementation=implementation, factorization=factorization)

    conv_dense = SphericalConv(
        3, 3, n_modes, n_layers=1, bias=False, implementation='reconstructed', factorization=None)

    conv_dense.weight[0] = FactorizedTensor.from_tensor(conv.weight[0].to_tensor(), rank=None, factorization='ComplexDense')
    x = torch.randn(2, 3, *(12, 12))

    res_dense = conv_dense(x)
    res = conv(x)
    res_shape = res.shape

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