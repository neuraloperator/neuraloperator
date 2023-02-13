import pytest
import torch
from tltorch import FactorizedTensor
from neuralop.models.fno_block import FactorizedSpectralConv2d, FactorizedSpectralConv1d, FactorizedSpectralConv


@pytest.mark.parametrize('factorization', ['ComplexDense', 'ComplexCP', 'ComplexTucker', 'ComplexTT'])
@pytest.mark.parametrize('implementation', ['factorized', 'reconstructed'])
def test_FactorizedSpectralConv(factorization, implementation):
    """Test for FactorizedSpectralConv of any order
    
    Verifies that a dense conv and factorized conv with the same weight produce the same output
    """
    modes = (4, 5, 3, 3)

    # Test for Conv1D to Conv4D
    for dim in [1, 2, 3, 4]:
        conv = FactorizedSpectralConv(
            10, 10, modes[:dim], n_layers=1, scale='auto', bias=False, implementation=implementation, factorization=factorization)

        conv_dense = FactorizedSpectralConv(
            10, 10, modes[:dim], n_layers=1, scale='auto', bias=False, implementation='reconstructed', factorization=None)

        for i in range(2**(dim-1)):
            conv_dense.weight[i] = FactorizedTensor.from_tensor(conv.weight[i].to_tensor(), rank=None, factorization='ComplexDense')

        x = torch.randn(2, 10, *(12, )*dim)

        res_dense = conv_dense(x)
        res = conv(x)

        torch.testing.assert_close(res_dense, res)


@pytest.mark.parametrize('factorization', ['ComplexCP', 'ComplexTucker'])
@pytest.mark.parametrize('implementation', ['factorized', 'reconstructed'])
def test_FactorizedSpectralConv2D(factorization, implementation):
    """Compare generic FactorizedSPectralConv with hand written FactorizedSpectralConv2D
    
    Verifies that a dense conv and factorized conv with the same weight produce the same output
    Note that this implies the order in which the conv is done in the manual implementation matches the automatic one, 
    take with a grain of salt
    """
    conv = FactorizedSpectralConv(
        10, 11, (4, 5), n_layers=1, scale='auto', bias=False, implementation=implementation, factorization=factorization
    )

    conv_dense = FactorizedSpectralConv2d(
        10, 11, 4, 5, n_layers=1, scale='auto', bias=False, implementation='reconstructed', factorization=None
    )
    for i, w in enumerate(conv.weight):
        rec = w.to_tensor()
        dtype = rec.dtype
        assert dtype == torch.cfloat
        conv_dense.weight[i] = FactorizedTensor.from_tensor(rec, rank=None, factorization='ComplexDense')

    x = torch.randn(2, 10, 12, 12)
    res_dense = conv_dense(x)
    res = conv(x)
    torch.testing.assert_close(res_dense, res)


@pytest.mark.parametrize('factorization', ['ComplexCP', 'ComplexTucker'])
@pytest.mark.parametrize('implementation', ['factorized', 'reconstructed'])
def test_FactorizedSpectralConv1D(factorization, implementation):
    """Test for FactorizedSpectralConv1D
    
    Verifies that a dense conv and factorized conv with the same weight produce the same output
    """
    conv = FactorizedSpectralConv(
        10, 11, (5,), n_layers=1, scale='auto', bias=False, implementation=implementation, factorization=factorization
    )

    conv_dense = FactorizedSpectralConv1d(
        10, 11, 5, n_layers=1, scale='auto', bias=False, implementation='reconstructed', factorization=None
    )
    for i, w in enumerate(conv.weight):
        rec = w.to_tensor()
        dtype = rec.dtype
        assert dtype == torch.cfloat
        conv_dense.weight[i] = FactorizedTensor.from_tensor(rec, rank=None, factorization='ComplexDense')

    x = torch.randn(2, 10, 12)
    res_dense = conv_dense(x)
    res = conv(x)
    torch.testing.assert_close(res_dense, res)
