import math
import torch
import pytest
from torch.testing import assert_close

from ..data_losses import LpLoss, H1Loss, HdivLoss
from ..differentiation import central_diff_1d, central_diff_2d, central_diff_3d, non_uniform_fd, FiniteDiff, FourierDiff
from neuralop.layers.embeddings import regular_grid_nd


def test_lploss():
    l2_2d_mean = LpLoss(d=2, p=2, reduction='mean', measure=1.)
    l2_2d_sum = LpLoss(d=2, p=2, reduction='sum', measure=1.)
    x = torch.randn(10, 1, 4, 4)

    abs_0 = l2_2d_mean.abs(x,x)
    assert abs_0.item() == 0.

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    # L2 w/out normalizing constant
    # sum of items in each element in ones is 16
    # norm is 4
    mean_abs_l2_err = l2_2d_mean.abs(zeros, ones)
    assert mean_abs_l2_err.item() == 1.

    sum_abs_l2_err = l2_2d_sum.abs(zeros, ones)
    assert sum_abs_l2_err.item() == 10.

    # Test quadrature weights: for spatial dims, each weight should
    # be 1/the corresponding size in x, so that after applying weights,
    # lploss constitutes an average over spatial dims. 
    d = 2
    default_quad_weights = l2_2d_mean.uniform_quadrature(x)
    assert default_quad_weights[-d:] == [1 / s for s in x.shape[-d:]]
    

    # Sanity check: ensure that both sum and mean reduction sum-reduce over spatial dims
    x = torch.arange(1,5) * torch.ones(4,4)
    flipped_x = x.T.unsqueeze(0).unsqueeze(0)
    x = x.unsqueeze(0).unsqueeze(0)

    zeros = torch.zeros_like(x)

    # batch and channel size of 1. squared diff should be 120, avg over spatial dims 7.5
    mean_err = l2_2d_mean.abs(flipped_x, zeros)
    sum_err = l2_2d_sum.abs(x, zeros)

    assert mean_err == sum_err
    assert_close(mean_err, torch.sqrt(torch.tensor(7.5)))



def test_h1loss():
    h1 = H1Loss(d=2, reduction='mean')
    x = torch.randn(10, 4, 4)

    abs_0 = h1.abs(x,x)
    assert abs_0.item() == 0.

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    # H1 w/o normalizing constant, 
    # finite-difference derivatives of both sides are zero
    # sum of items in each element in ones is 16
    # norm is 1 averaged in space
    mean_abs_h1 = h1.abs(zeros, ones)
    assert mean_abs_h1.item() == 1.

def test_hdivloss():
    hdiv = HdivLoss(d=2, reduction='mean')
    x = torch.randn(10, 4, 4)

    abs_0 = hdiv.abs(x,x)
    assert abs_0.item() == 0.

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    # Hdiv w/o normalizing constant, 
    # finite-difference derivatives of both sides are zero
    # sum of items in each element in ones is 16
    # norm is 1 averaged in space
    mean_abs_hdiv = hdiv.abs(zeros, ones)
    assert mean_abs_hdiv.item() == 1.



@pytest.mark.parametrize("dim", [1, 2, 3])
def test_central_diff(dim: int):
    
    if dim == 1:
        # assert f(x) = x
        # has derivative 1 everywhere when boundaries are fixed
        x = torch.arange(10)
        dx = central_diff_1d(x, h=1., periodic_in_x=False)
        assert_close(dx,torch.ones_like(dx))
        
    if dim == 2:

        grid = regular_grid_nd(resolutions=[10,10], grid_boundaries=[[0,10]] * 2)
        x = torch.stack(grid, dim=0)
        dx, dy = central_diff_2d(x, h=1., periodic_in_x=False, periodic_in_y=False)
        # pos encoding A[:,i,j] = [xi, yj]

        # dx[:,i,j] = f(x_i, y_j) vector valued <fx, fy>
        # dfx(coords) == 1s
        
        assert_close(dx[0], torch.ones_like(dx[0]))
        assert_close(dx[1], torch.zeros_like(dx[1]))
        
        assert_close(dy[0], torch.zeros_like(dy[0]))
        assert_close(dy[1], torch.ones_like(dy[1]))

    if dim == 3:
        grid = regular_grid_nd(resolutions=[10,10,10], grid_boundaries=[[0,10]] * 3)
        x = torch.stack(grid, dim=0)
        # pos encoding A[:,i,j,k] = [xi, yj, zk]
        dx, dy, dz = central_diff_3d(x, h=1., periodic_in_x=False, periodic_in_y=False, periodic_in_z=False)
        # dx[:,i,j,k] = f(x_i, y_j, z_k) vector valued <fx, fy, fz>
        # dfx(coords) == 1s
        
        assert_close(dx[0], torch.ones_like(dx[0]))
        assert_close(dx[1], torch.zeros_like(dx[1]))
        assert_close(dx[2], torch.zeros_like(dx[1]))
        
        assert_close(dy[0], torch.zeros_like(dy[0]))
        assert_close(dy[1], torch.ones_like(dy[1]))
        assert_close(dy[2], torch.zeros_like(dy[2]))

        assert_close(dz[0], torch.zeros_like(dz[0]))
        assert_close(dz[1], torch.zeros_like(dz[1]))
        assert_close(dz[2], torch.ones_like(dz[2]))



@pytest.mark.parametrize("dim", [1, 2])
def test_nonuniform_fd(dim: int):
    
    if dim == 1:
        x = torch.sort(torch.rand(256))[0].unsqueeze(1)
        f = torch.exp(3*x) + torch.sin(10*x) - x**2
        df_ref = (3*torch.exp(3*x) + 10*torch.cos(10*x) - 2 * x).squeeze()
        df_dx = non_uniform_fd(x, f.squeeze(), num_neighbors=3, derivative_indices=[0], regularize_lstsq=False)[0]
        
        l2 = LpLoss(d=1, p=2, reduction='mean', measure=1.)
        assert l2.rel(df_ref, df_dx).item() < 5e-2
        
        # # Plot to check visually
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10,5))
        # plt.plot(x.detach().numpy().squeeze(), df_ref.detach().numpy(), 'r', linewidth=0.8, label='Reference')
        # plt.plot(x.detach().numpy(), df_dx.detach().numpy(), 'b', linewidth=0.8, label='FD')
        # plt.legend()
        # plt.xlabel(r'$x$')
        # plt.ylabel(r'$df/dx$')
        # plt.savefig('non_uniform_fd_1D.pdf')
    
    if dim == 2:
        num_points = 128
        x = torch.linspace(-1, 1, num_points, dtype=torch.float64)
        y = torch.linspace(-1, 1, num_points, dtype=torch.float64)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        f = torch.exp(Y) + 0.1 * torch.sin(10 * X) - (X**2) * (Y**2)
        dfdx_ref =  torch.cos(10*X) - 2*X*(Y**2)
        dfdy_ref = torch.exp(Y) - 2*(X**2)*Y
        df = non_uniform_fd(points, f.flatten(), num_neighbors=5, derivative_indices=[0,1], regularize_lstsq=True)
        df_dx = df[0].reshape(num_points, num_points)
        df_dy = df[1].reshape(num_points, num_points)

        l2 = LpLoss(d=2, p=2, reduction='mean', measure=1.)
        assert l2.rel(dfdx_ref, df_dx).item() < 5e-2
        assert l2.rel(dfdy_ref, df_dy).item() < 5e-2
        
        # # Plot to check visually
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(7, 7))
        # plt.subplot(2, 2, 1)
        # img1 = plt.imshow(df_dx.detach().numpy())
        # plt.colorbar(img1, shrink=0.75)  
        # plt.title(r'$df/dx$')
        # plt.xticks([])  
        # plt.yticks([]) 
        # plt.subplot(2, 2, 2)
        # img2 = plt.imshow(dfdx_ref.detach().numpy())
        # plt.colorbar(img2, shrink=0.75)
        # plt.title(r'$df/dx$_ref')
        # plt.xticks([])
        # plt.yticks([])
        # plt.subplot(2, 2, 3)
        # img3 = plt.imshow(df_dy.detach().numpy())
        # plt.colorbar(img3, shrink=0.75)
        # plt.title(r'$df/dy$')
        # plt.xticks([])  
        # plt.yticks([]) 
        # plt.subplot(2, 2, 4)
        # img4 = plt.imshow(dfdy_ref.detach().numpy().reshape(num_points, num_points))
        # plt.colorbar(img4, shrink=0.75)
        # plt.title(r'$df/dy$_ref')
        # plt.xticks([])
        # plt.yticks([])
        # plt.tight_layout()
        # plt.savefig('non_uniform_fd_2D.pdf')


@pytest.mark.parametrize("periodic_x", [True, False])
@pytest.mark.parametrize("periodic_y", [True, False])
def test_finite_diff_2d(periodic_x, periodic_y):
    """Test the FiniteDiff class with various boundary conditions for 2D."""
    
    # Create a 2D test function: f(x,y) = x^2 + y^2
    nx, ny = 32, 32
    x = torch.linspace(0, 2*torch.pi, nx, dtype=torch.float64)
    y = torch.linspace(0, 2*torch.pi, ny, dtype=torch.float64)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    f = X**2 + Y**2
    
    # Initialize FiniteDiff with dim=2 and specified boundary conditions
    fd2d = FiniteDiff(dim=2, h=(0.2, 0.2), periodic_in_x=periodic_x, periodic_in_y=periodic_y)
    
    # Test first order derivatives
    df_dx = fd2d.dx(f)
    df_dy = fd2d.dy(f)
    
    # Test second order derivatives
    d2f_dx2 = fd2d.dx(f, order=2)
    d2f_dy2 = fd2d.dy(f, order=2)
    
    # Test laplacian
    laplacian = fd2d.laplacian(f)
    
    # Test gradient
    gradient = fd2d.gradient(f)
    
    # Test with vector field for divergence and curl
    # Create a simple vector field: u = [sin(x), cos(y)]
    u1 = torch.sin(X)
    u2 = torch.cos(Y)
    u_vector = torch.stack([u1, u2], dim=0)
    
    divergence = fd2d.divergence(u_vector)
    curl = fd2d.curl(u_vector)
    
    # Basic shape assertions
    assert df_dx.shape == f.shape
    assert df_dy.shape == f.shape
    assert d2f_dx2.shape == f.shape
    assert d2f_dy2.shape == f.shape
    assert laplacian.shape == f.shape
    assert gradient.shape == (2, nx, ny)
    assert divergence.shape == (nx, ny)
    assert curl.shape == (nx, ny)
    
    # Test that laplacian equals sum of second derivatives
    assert_close(laplacian, d2f_dx2 + d2f_dy2)


@pytest.mark.parametrize("periodic_x", [True, False])
def test_finite_diff_1d(periodic_x):
    """Test the FiniteDiff class with various boundary conditions for 1D."""
    
    # Create a 1D test function: f(x) = cos(x) - x
    nx = 64
    x = torch.linspace(0, 2*torch.pi, nx, dtype=torch.float64)
    f = torch.cos(x) - x
    
    # Initialize FiniteDiff with dim=1 and specified boundary conditions
    fd1d = FiniteDiff(dim=1, h=2*torch.pi/nx, periodic_in_x=periodic_x)
    
    # Test first order derivatives
    df_dx = fd1d.dx(f)
    
    # Test second order derivatives
    d2f_dx2 = fd1d.dx(f, order=2)
    
    # Basic shape assertions
    assert df_dx.shape == f.shape
    assert d2f_dx2.shape == f.shape
    
    # Test that first derivative is approximately correct for f(x) = cos(x) - x
    # df/dx ≈ -sin(x) - 1 for f(x) = cos(x) - x
    if not periodic_x:
        # For non-periodic, interior points should be close to expected value
        expected_df_dx = -torch.sin(x[2:-2]) - 1.0
        assert_close(df_dx[2:-2], expected_df_dx, atol=0.01, rtol=0.1)


@pytest.mark.parametrize("periodic_x", [True, False])
@pytest.mark.parametrize("periodic_y", [True, False])
@pytest.mark.parametrize("periodic_z", [True, False])
def test_finite_diff_3d(periodic_x, periodic_y, periodic_z):
    """Test the FiniteDiff class with various boundary conditions for 3D."""
    
    # Create a 3D test function: f(x,y,z) = x^2 + y^2 + z^2
    nx, ny, nz = 16, 16, 16
    x = torch.linspace(0, 2*torch.pi, nx, dtype=torch.float64)
    y = torch.linspace(0, 2*torch.pi, ny, dtype=torch.float64)
    z = torch.linspace(0, 2*torch.pi, nz, dtype=torch.float64)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    f = X**2 + Y**2 + Z**2
    
    # Initialize FiniteDiff with dim=3 and specified boundary conditions
    fd3d = FiniteDiff(dim=3, h=(0.4, 0.4, 0.4), periodic_in_x=periodic_x, periodic_in_y=periodic_y, periodic_in_z=periodic_z)
    
    # Test first order derivatives
    df_dx = fd3d.dx(f)
    df_dy = fd3d.dy(f)
    df_dz = fd3d.dz(f)
    
    # Test second order derivatives
    d2f_dx2 = fd3d.dx(f, order=2)
    d2f_dy2 = fd3d.dy(f, order=2)
    d2f_dz2 = fd3d.dz(f, order=2)
    
    # Test laplacian
    laplacian = fd3d.laplacian(f)
    
    # Test gradient
    gradient = fd3d.gradient(f)
    
    # Test with vector field for divergence and curl
    # Create a simple vector field: u = [sin(x), cos(y), sin(z)]
    u1 = torch.sin(X)
    u2 = torch.cos(Y)
    u3 = torch.sin(Z)
    u_vector = torch.stack([u1, u2, u3], dim=0)
    
    divergence = fd3d.divergence(u_vector)
    curl = fd3d.curl(u_vector)
    
    # Basic shape assertions
    assert df_dx.shape == f.shape
    assert df_dy.shape == f.shape
    assert df_dz.shape == f.shape
    assert d2f_dx2.shape == f.shape
    assert d2f_dy2.shape == f.shape
    assert d2f_dz2.shape == f.shape
    assert laplacian.shape == f.shape
    assert gradient.shape == (3, nz, ny, nx)
    assert divergence.shape == (nx, ny, nz)
    assert curl.shape == (3, nz, ny, nx)
    
    # Test that laplacian equals sum of second derivatives
    assert_close(laplacian, d2f_dx2 + d2f_dy2 + d2f_dz2)


@pytest.mark.parametrize("periodic", [True, False])
def test_fourier_diff(periodic: bool):

    if periodic:

        ## Test on periodic functions without Fourier continuation
        # Consider sin(x) and cos(x)
        L = 2*torch.pi
        x = torch.linspace(0, L, 101)[:-1]
        f = torch.stack([torch.sin(x), torch.cos(x)], dim=0)
        
        # Use FourierDiff class
        fd1d = FourierDiff(dim=1, L=L, use_fc=False)
        derivatives = fd1d.compute_multiple_derivatives(f, [1, 2, 3])
        dfdx, df2dx2, df3dx3 = derivatives
        
        assert f.shape == dfdx.shape == df2dx2.shape == df3dx3.shape

        
    else: 
    
        ## Test on non-periodic functions using Fourier continuation
        # Consider sin(16*x)-cos(8*x) and exp(-0.8x)
        L = 2*torch.pi
        x = torch.linspace(0, L, 101)[:-1]    
        f = torch.stack([torch.sin(3*x) - torch.cos(x), torch.exp(-0.8*x)+torch.sin(x)], dim=0)
        
        # Use FourierDiff class with Fourier continuation
        fd1d = FourierDiff(dim=1, L=L, use_fc='Legendre', fc_degree=4, fc_n_additional_pts=30)
        derivatives = fd1d.compute_multiple_derivatives(f, [1, 2])
        dfdx, df2dx2 = derivatives

        assert f.shape == dfdx.shape == df2dx2.shape
   

@pytest.mark.parametrize("use_fc", [False, 'Legendre'])
def test_fourier_diff_2d(use_fc):
    
    if use_fc:
        fd2d = FourierDiff(dim=2, L=(2*torch.pi, 2*torch.pi), use_fc=use_fc, fc_degree=4, fc_n_additional_pts=20)
    else:
        fd2d = FourierDiff(dim=2, L=(2*torch.pi, 2*torch.pi))
    
    # Create a 2D periodic function: sin(x) * cos(y)
    L_x, L_y = 2*torch.pi, 2*torch.pi
    nx, ny = 64, 64
    x = torch.linspace(0, L_x, nx, dtype=torch.float64)
    y = torch.linspace(0, L_y, ny, dtype=torch.float64)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Test function: f(x,y) = sin(x) * cos(y) for periodic, exp(-x) * sin(y) for non-periodic
    if use_fc:
        f = torch.exp(-X) * torch.sin(Y)  # Non-periodic function
    else:
        f = torch.sin(X) * torch.cos(Y)  # Periodic function
    
    # Test partial derivatives
    df_dx_computed = fd2d.dx(f)
    df_dy_computed = fd2d.dy(f)
    
    # Test higher order derivatives
    d2f_dx2_computed = fd2d.dx(f, order=2)
    d2f_dy2_computed = fd2d.dy(f, order=2)
    
    # Test laplacian
    laplacian_computed = fd2d.laplacian(f)
    
    # Test multiple derivatives computation
    derivatives = fd2d.compute_multiple_derivatives(f, [(1, 0), (0, 1), (2, 0), (0, 2)])
    df_dx_multi, df_dy_multi, d2f_dx2_multi, d2f_dy2_multi = derivatives
    
    # Test gradient
    gradient = fd2d.gradient(f)
    assert gradient.shape == (2, nx, ny)
    
    # Test divergence with a vector field
    # Create a vector field: u = [sin(x), cos(y)]
    u1 = torch.sin(X)
    u2 = torch.cos(Y)
    u_vector = torch.stack([u1, u2], dim=0)
    divergence = fd2d.divergence(u_vector)
    
    # Test curl with a vector field
    curl = fd2d.curl(u_vector)
    
    # Basic shape assertions - just check that derivatives have correct shapes
    assert df_dx_computed.shape == f.shape
    assert df_dy_computed.shape == f.shape
    assert d2f_dx2_computed.shape == f.shape
    assert d2f_dy2_computed.shape == f.shape
    assert laplacian_computed.shape == f.shape
    assert len(derivatives) == 4
    assert gradient.shape == (2, nx, ny)
    assert divergence.shape == (nx, ny)
    assert curl.shape == (nx, ny)
    
@pytest.mark.parametrize("use_fc", [False, 'Gram'])
def test_fourier_diff_3d(use_fc):
    """Test the FourierDiff class with various scenarios."""
    
    # Test basic functionality
    if use_fc:
        fd3d = FourierDiff(dim=3, L=(2*torch.pi, 2*torch.pi, 2*torch.pi), use_fc=use_fc, fc_degree=4, fc_n_additional_pts=50)
    else:
        fd3d = FourierDiff(dim=3, L=(2*torch.pi, 2*torch.pi, 2*torch.pi))
    
    # Create a 3D periodic function: sin(x) * cos(y) * sin(z)
    L_x, L_y, L_z = 2*torch.pi, 2*torch.pi, 2*torch.pi
    nx, ny, nz = 32, 32, 32
    x = torch.linspace(0, L_x, nx, dtype=torch.float64)
    y = torch.linspace(0, L_y, ny, dtype=torch.float64)
    z = torch.linspace(0, L_z, nz, dtype=torch.float64)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Test function: f(x,y,z) = sin(x) * cos(y) * sin(z) for periodic, exp(-x) * sin(y) * cos(z) for non-periodic
    if use_fc:
        f = torch.sin(X) * torch.cos(Y) * torch.sin(Z)  # Non-periodic function
    else:
        f = torch.sin(X) * torch.cos(Y) * torch.sin(Z)  # Periodic function
    
    # Test partial derivatives
    df_dx_computed = fd3d.dx(f)
    df_dy_computed = fd3d.dy(f)
    df_dz_computed = fd3d.dz(f)
    
    # Test higher order derivatives
    d2f_dx2_computed = fd3d.dx(f, order=2)
    d2f_dy2_computed = fd3d.dy(f, order=2)
    d2f_dz2_computed = fd3d.dz(f, order=2)
    d2f_dxdy_computed = fd3d.dx(fd3d.dy(f))
    d2f_dxdz_computed = fd3d.dx(fd3d.dz(f))
    d2f_dydz_computed = fd3d.dy(fd3d.dz(f))
    
    # Test laplacian
    laplacian_computed = fd3d.laplacian(f)
    
    # Test multiple derivatives computation
    derivatives = fd3d.compute_multiple_derivatives(f, [(1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)])
    df_dx_multi, df_dy_multi, df_dz_multi, d2f_dx2_multi, d2f_dy2_multi, d2f_dz2_multi, d2f_dxdy_multi, d2f_dxdz_multi, d2f_dydz_multi = derivatives
    
    # Test gradient
    gradient = fd3d.gradient(f)
    assert gradient.shape == (3, nz, ny, nx)
    
    # Test divergence with a vector field
    # Create a vector field: u = [sin(x), cos(y), sin(z)]
    u1 = torch.sin(X)
    u2 = torch.cos(Y)
    u3 = torch.sin(Z)
    u_vector = torch.stack([u1, u2, u3], dim=0)
    divergence = fd3d.divergence(u_vector)
    
    # Test curl with a vector field
    curl = fd3d.curl(u_vector)
    
    # Basic shape assertions - just check that derivatives have correct shapes
    assert df_dx_computed.shape == f.shape
    assert df_dy_computed.shape == f.shape
    assert df_dz_computed.shape == f.shape
    assert d2f_dx2_computed.shape == f.shape
    assert d2f_dy2_computed.shape == f.shape
    assert d2f_dz2_computed.shape == f.shape
    assert d2f_dxdy_computed.shape == f.shape
    assert d2f_dxdz_computed.shape == f.shape
    assert d2f_dydz_computed.shape == f.shape
    assert laplacian_computed.shape == f.shape
    assert len(derivatives) == 9
    assert gradient.shape == (3, nz, ny, nx)
    assert divergence.shape == (nx, ny, nz)
    assert curl.shape == (3, nz, ny, nx)
   