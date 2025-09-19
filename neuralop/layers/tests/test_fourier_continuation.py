import torch
import pytest
from ..fourier_continuation import FCLegendre, FCGram

@pytest.mark.parametrize("FC_class", [FCLegendre, FCGram])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("d", [3, 4])
def test_fourier_continuation(FC_class, dim: int, d: int):
    
    if dim == 1:
        
        batch_size = 3
        length_signal = 101
        add_pts = 50

        x = torch.linspace(0, 1, length_signal).repeat(batch_size,1)
        f = torch.sin(16 * x) - torch.cos(8 * x)
        
        Extension = FC_class(d=d, n_additional_pts=add_pts)

        f_extend = Extension(f, dim=1)

        # Check shape
        assert f_extend.shape[-1] == f.shape[-1] + add_pts
        # Check values of original signal
        torch.testing.assert_close(f, f_extend[...,add_pts//2:-add_pts//2])

    if dim == 2:
        
        batch_size = 3
        length_signal = 101
        add_pts = 50

        x = torch.linspace(0, 1, length_signal).view(1, length_signal, 1).repeat(batch_size, 1, length_signal)
        y = torch.linspace(0, 1, length_signal).view(1, 1, length_signal).repeat(batch_size, length_signal, 1)
        f = torch.sin(12 * x)  - torch.cos(14 * y) + 3*x*y
        
        Extension = FC_class(d=d, n_additional_pts=add_pts)

        f_extend = Extension(f, dim=2)

        # Check shape
        assert f_extend.shape[-1] == f.shape[-1] + add_pts
        assert f_extend.shape[-2] == f.shape[-2] + add_pts
        # Check values of original signal
        torch.testing.assert_close(f, f_extend[...,add_pts//2:-add_pts//2, add_pts//2:-add_pts//2])

    if dim == 3:
        
        batch_size = 2
        length_signal = 81
        add_pts = 50

        # Create 3D grid
        x = torch.linspace(0, 1, length_signal).view(1, length_signal, 1, 1).repeat(batch_size, 1, length_signal, length_signal)
        y = torch.linspace(0, 1, length_signal).view(1, 1, length_signal, 1).repeat(batch_size, length_signal, 1, length_signal)
        z = torch.linspace(0, 1, length_signal).view(1, 1, 1, length_signal).repeat(batch_size, length_signal, length_signal, 1)

        # Create 3D function
        f = torch.exp(-2*z) + 2*z*x + torch.sin(12*x*y) + y*torch.sin(10*y*z) 

        Extension = FC_class(d=d, n_additional_pts=add_pts)

        f_extend = Extension(f, dim=3)

        # Check shape
        assert f_extend.shape[-1] == f.shape[-1] + add_pts
        assert f_extend.shape[-2] == f.shape[-2] + add_pts
        assert f_extend.shape[-3] == f.shape[-3] + add_pts
        # Check values of original signal
        torch.testing.assert_close(f, f_extend[...,add_pts//2:-add_pts//2, add_pts//2:-add_pts//2, add_pts//2:-add_pts//2])


@pytest.mark.parametrize("FC_class", [FCLegendre, FCGram])
@pytest.mark.parametrize("dim_tuple", [(1,), (2,), (1, 2), (1, 3), (2, 3), (-1,), (-2,), (-1, -2), (-1, -3), (1, -1)])
@pytest.mark.parametrize("d", [3, 4])
def test_fourier_continuation_tuple_dims(FC_class, dim_tuple, d: int):
    """
    Test Fourier continuation with tuple dimensions.
    
    Tests various combinations of positive and negative axis indices
    to ensure the extension works correctly along specific axes.
    """
    batch_size = 2
    add_pts = 30
    
    # Create a 4D tensor for comprehensive testing
    x = torch.randn(batch_size, 20, 25, 30)
    
    Extension = FC_class(d=d, n_additional_pts=add_pts)
    
    # Extend along specified dimensions
    f_extend = Extension(x, dim=dim_tuple)
    
    # Calculate expected shape
    expected_shape = list(x.shape)
    for dim in dim_tuple:
        # Convert negative indices to positive
        if dim < 0:
            dim = x.ndim + dim
        expected_shape[dim] += add_pts
    
    # Check shape
    assert f_extend.shape == tuple(expected_shape), f"Expected shape {tuple(expected_shape)}, got {f_extend.shape}"
    
    # Test restrict functionality
    f_restricted = Extension.restrict(f_extend, dim=dim_tuple)
    
    # Check that restrict returns original shape
    assert f_restricted.shape == x.shape, f"Restrict failed: expected {x.shape}, got {f_restricted.shape}"
    
    # Check that restrict returns original values
    torch.testing.assert_close(x, f_restricted, msg="Restrict did not return original values")


@pytest.mark.parametrize("FC_class", [FCLegendre, FCGram])
def test_fourier_continuation_edge_cases(FC_class):
    """
    Test edge cases for tuple dimension extensions.
    """
    add_pts = 20
    d = 3
    
    # Test with single dimension tuple
    x = torch.randn(3, 15, 20)
    Extension = FC_class(d=d, n_additional_pts=add_pts)
    
    # Test single positive dimension
    f_extend = Extension(x, dim=(1,))
    expected_shape = (3, 15 + add_pts, 20)
    assert f_extend.shape == expected_shape
    
    # Test single negative dimension
    f_extend_neg = Extension(x, dim=(-1,))
    expected_shape_neg = (3, 15, 20 + add_pts)
    assert f_extend_neg.shape == expected_shape_neg
    
    # Test multiple dimensions
    f_extend_multi = Extension(x, dim=(1, 2))
    expected_shape_multi = (3, 15 + add_pts, 20 + add_pts)
    assert f_extend_multi.shape == expected_shape_multi
    
    # Test restrict with multiple dimensions
    f_restricted = Extension.restrict(f_extend_multi, dim=(1, 2))
    assert f_restricted.shape == x.shape
    torch.testing.assert_close(x, f_restricted)


@pytest.mark.parametrize("FC_class", [FCLegendre, FCGram])
def test_fourier_continuation_high_dimensional(FC_class):
    """
    Test Fourier continuation with high-dimensional tensors.
    """
    add_pts = 20
    d = 3
    
    # Test 5D tensor
    x_5d = torch.randn(2, 10, 12, 15, 18)
    Extension = FC_class(d=d, n_additional_pts=add_pts)
    
    # Test extending along middle dimensions
    f_extend = Extension(x_5d, dim=(2, 4))
    expected_shape = (2, 10, 12 + add_pts, 15, 18 + add_pts)
    assert f_extend.shape == expected_shape
    
    # Test restrict
    f_restricted = Extension.restrict(f_extend, dim=(2, 4))
    assert f_restricted.shape == x_5d.shape
    torch.testing.assert_close(x_5d, f_restricted)
    
    # Test extending along all spatial dimensions
    f_extend_all = Extension(x_5d, dim=(1, 2, 3, 4))
    expected_shape_all = (2, 10 + add_pts, 12 + add_pts, 15 + add_pts, 18 + add_pts)
    assert f_extend_all.shape == expected_shape_all
    
    # Test restrict all
    f_restricted_all = Extension.restrict(f_extend_all, dim=(1, 2, 3, 4))
    assert f_restricted_all.shape == x_5d.shape
    torch.testing.assert_close(x_5d, f_restricted_all)
