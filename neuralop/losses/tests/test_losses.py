import math
import torch
from torch.testing import assert_close

from ..data_losses import LpLoss, H1Loss, HdivLoss
from ..finite_diff import central_diff_1d, central_diff_2d, central_diff_3d, non_uniform_fd
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


def test_central_diff1d():
    # assert f(x) = x
    # has derivative 1 everywhere when boundaries are fixed
    x = torch.arange(10)
    dx = central_diff_1d(x, fix_x_bnd=True, h=1.)
    assert_close(dx,torch.ones_like(dx))

def test_central_diff2d():
    grid = regular_grid_nd(resolutions=[10,10], grid_boundaries=[[0,10]] * 2)
    x = torch.stack(grid, dim=0)
    dx, dy = central_diff_2d(x, fix_x_bnd=True, fix_y_bnd=True, h=1.)
    # pos encoding A[:,i,j] = [xi, yj]

    # dx[:,i,j] = f(x_i, y_j) vector valued <fx, fy>
    # dfx(coords) == 1s
    
    assert_close(dx[0], torch.ones_like(dx[0]))
    assert_close(dx[1], torch.zeros_like(dx[1]))
    
    assert_close(dy[0], torch.zeros_like(dy[0]))
    assert_close(dy[1], torch.ones_like(dy[1]))

def test_central_diff3d():
    grid = regular_grid_nd(resolutions=[10,10,10], grid_boundaries=[[0,10]] * 3)
    x = torch.stack(grid, dim=0)
    # pos encoding A[:,i,j,k] = [xi, yj, zk]
    dx, dy, dz = central_diff_3d(x, fix_x_bnd=True, fix_y_bnd=True, fix_z_bnd=True, h=1.)
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


def test_nonuniform_fd_1d():
    
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
    
    
def test_nonuniform_fd_2d():
    
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