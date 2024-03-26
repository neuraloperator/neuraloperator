import pytest
import torch
from tltorch import FactorizedTensor
from ..spectral_convolution import (SpectralConv3d, SpectralConv2d,
                                       SpectralConv1d, SpectralConv)
# from ..cp import (SpectralConv3d, SpectralConv2d,
#                                        SpectralConv1d, SpectralConv)



@pytest.mark.parametrize('factorization', ['ComplexDense', 'ComplexCP', 'ComplexTucker', 'ComplexTT'])
@pytest.mark.parametrize('implementation', ['factorized', 'reconstructed'])
def test_SpectralConv(factorization, implementation):
    """Test for SpectralConv of any order
    
    Compares Factorized and Dense convolution output
    Verifies that a dense conv and factorized conv with the same weight produce the same output

    Checks the output size

    Verifies that dynamically changing the number of Fourier modes doesn't break the conv
    """
    modes = (10, 8, 6, 6)
    incremental_modes = (6, 6, 4, 4)

    # Test for Conv1D to Conv4D
    for dim in [1, 2, 3, 4]:
        conv = SpectralConv(
            3, 3, modes[:dim], n_layers=1, bias=False, implementation=implementation, factorization=factorization)

        conv_dense = SpectralConv(
            3, 3, modes[:dim], n_layers=1, bias=False, implementation='reconstructed', factorization=None)

        conv_dense.weight[0] = FactorizedTensor.from_tensor(conv.weight[0].to_tensor(), rank=None, factorization='ComplexDense')

        x = torch.randn(2, 3, *(12, )*dim)

        res_dense = conv_dense(x)
        res = conv(x)
        res_shape = res.shape

        torch.testing.assert_close(res_dense, res)

        # Dynamically reduce the number of modes in Fourier space
        conv.n_modes = incremental_modes[:dim]
        res = conv(x)
        assert res_shape == res.shape

        # Downsample outputs
        block = SpectralConv(
            3, 4, modes[:dim], n_layers=1, output_scaling_factor=0.5)
    
        x = torch.randn(2, 3, *(12, )*dim)
        res = block(x)
        assert(list(res.shape[2:]) == [12//2]*dim)
        
        # Upsample outputs
        block = SpectralConv(
            3, 4, modes[:dim], n_layers=1, output_scaling_factor=2)
    
        x = torch.randn(2, 3, *(12, )*dim)
        res = block(x)
        assert res.shape[1] == 4 # Check out channels
        assert(list(res.shape[2:]) == [12*2]*dim)



def test_SpectralConv_output_scaling_factor():
    """Test SpectralConv with upsampled or downsampled outputs
    """
    modes = (4, 4, 4, 4)
    size = [6]*4
    for dim in [1, 2, 3, 4]:
        # Downsample outputs
        conv = SpectralConv(
            3, 3, modes[:dim], n_layers=1, output_scaling_factor=0.5)
    
        x = torch.randn(2, 3, *size[:dim])
        res = conv(x)
        assert(list(res.shape[2:]) == [m//2 for m in size[:dim]])
        
        # Upsample outputs
        conv = SpectralConv(
            3, 3, modes[:dim], n_layers=1, output_scaling_factor=2)
    
        x = torch.randn(2, 3, *size[:dim])
        res = conv(x)
        assert(list(res.shape[2:]) == [m*2 for m in size[:dim]])


@pytest.mark.parametrize('spatial_domain', ["real", "complex"])
def test_SpectralConv_complex(spatial_domain):
    """Test SpectralConv when input is real or complex
    """
    modes = (4, 4, 4, 4)
    size = [6]*4
    for dim in [1, 2, 3, 4]:
        conv = SpectralConv(3, 3, modes[:dim], n_layers=1, spatial_domain=spatial_domain)
    
        if spatial_domain == "real":
            input_type = torch.float
        elif spatial_domain == "complex":
            input_type = torch.cfloat

        x = torch.randn(2, 3, *size[:dim], dtype=input_type)
        res = conv(x)

        assert x.dtype == input_type
        assert res.dtype == input_type
        assert x.shape == res.shape


@pytest.mark.parametrize('factorization', ['ComplexCP', 'ComplexTucker'])
@pytest.mark.parametrize('implementation', ['factorized', 'reconstructed'])
def test_SpectralConv3D(factorization, implementation):
    """Compare generic SpectralConv with hand written SpectralConv2D
    
    Verifies that a dense conv and factorized conv with the same weight produce the same output
    Note that this implies the order in which the conv is done in the manual implementation matches the automatic one, 
    take with a grain of salt
    """
    conv = SpectralConv(
        3, 6, (4, 4, 3), n_layers=1, bias=False, implementation=implementation, factorization=factorization
    )

    conv_dense = SpectralConv3d(
        3, 6, (4, 4, 3), n_layers=1, bias=False, implementation='reconstructed', factorization=None
    )
    for i, w in enumerate(conv.weight):
        rec = w.to_tensor()
        dtype = rec.dtype
        assert dtype == torch.cfloat
        conv_dense.weight[i] = FactorizedTensor.from_tensor(rec, rank=None, factorization='ComplexDense')

    x = torch.randn(2, 3, 12, 12, 12)
    res_dense = conv_dense(x)
    res = conv(x)
    torch.testing.assert_close(res_dense, res)




@pytest.mark.parametrize('factorization', ['ComplexCP', 'ComplexTucker', 'ComplexDense'])
@pytest.mark.parametrize('implementation', ['factorized', 'reconstructed'])
def test_SpectralConv2D(factorization, implementation):
    """Compare generic SpectralConv with hand written SpectralConv2D
    
    Verifies that a dense conv and factorized conv with the same weight produce the same output
    Note that this implies the order in which the conv is done in the manual implementation matches the automatic one, 
    take with a grain of salt
    """
    conv = SpectralConv(
        10, 11, (4, 5), n_layers=1, bias=False, implementation=implementation, factorization=factorization
    )

    conv_dense = SpectralConv2d(
        10, 11, (4, 5), n_layers=1, bias=False, implementation='reconstructed', factorization=None
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
def test_SpectralConv1D(factorization, implementation):
    """Test for SpectralConv1D
    
    Verifies that a dense conv and factorized conv with the same weight produce the same output
    """
    conv = SpectralConv(
        10, 11, (5,), n_layers=1, bias=False, implementation=implementation, factorization=factorization
    )
    conv_dense = SpectralConv1d(
        10, 11, (5,), n_layers=1, bias=False, implementation='reconstructed', factorization=None
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
