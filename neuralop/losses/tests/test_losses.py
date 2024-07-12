import math
import torch

from ..data_losses import LpLoss, H1Loss, MSELoss

def test_lploss():
    l2_2d_mean = LpLoss(d=2, p=2, reductions='mean')
    l2_2d_sum = LpLoss(d=2, p=2, reductions='sum')
    x = torch.randn(10, 4, 4)

    abs_0 = l2_2d_mean.abs(x,x)
    assert abs_0.item() == 0.

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    # L2 w/out normalizing constant
    # sum of items in each element in ones is 16
    # norm is 4
    mean_abs_l2_err = l2_2d_mean.abs(zeros, ones, h=1.)
    assert mean_abs_l2_err.item() == 4.

    sum_abs_l2_err = l2_2d_sum.abs(zeros, ones, h=1.)
    assert sum_abs_l2_err.item() == 40.

    eps = 1e-7
    # L2 with default 1d normalizing constant 
    # result should be scaled by 2pi/(geometric mean of input dims= 4)
    mean_abs_l2_err = l2_2d_mean.abs(zeros, ones)
    assert mean_abs_l2_err.item() - (4. * math.pi / 2) <= eps

    sum_abs_l2_err = l2_2d_sum.abs(zeros, ones)
    assert sum_abs_l2_err.item() - 40. * math.pi / 2  <= eps

def test_h1loss():
    h1 = LpLoss(d=2, p=2, reductions='mean')
    x = torch.randn(10, 4, 4)

    abs_0 = h1.abs(x,x)
    assert abs_0.item() == 0.

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    # H1 w/out normalizing constant, 
    # finite-difference derivatives of both sides are zero
    # sum of items in each element in ones is 16
    # norm is 4
    mean_abs_h1 = h1.abs(zeros, ones, h=1.)
    assert mean_abs_h1.item() == 4.

def test_mseloss():
    mse_2d = MSELoss(reductions='sum')
    x = torch.randn(10, 4, 4)

    abs_0 = mse_2d(x,x)
    assert abs_0.item() == 0.

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    # all elem-wise differences are 1., squared and averaged = 1.
    # reduced by sum across batch = 10 * 1. = 10.
    mean_abs_mse = mse_2d(zeros, ones)
    assert mean_abs_mse.item() == 10.