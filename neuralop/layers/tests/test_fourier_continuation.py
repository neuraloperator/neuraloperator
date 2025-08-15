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
