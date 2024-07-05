import torch

from ..data_losses import LpLoss, MSELoss

def test_lploss():
    l2_2d_mean = LpLoss(d=2, p=2, reductions='mean')
    l2_2d_sum = LpLoss(d=2, p=2, reductions='sum')
    x = torch.randn(10, 4, 4)

    abs_0 = l2_2d_mean.abs(x,x)
    assert abs_0.item() == 0.

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    # sum of items in each element in ones is 16
    # norm is 4
    mean_abs_l2_err = l2_2d_mean.abs(zeros, ones, h=1.)
    assert mean_abs_l2_err.item() == 4.

    sum_abs_l2_err = l2_2d_sum.abs(zeros, ones, h=1.)
    assert sum_abs_l2_err.item() == 40.

def test_lploss():
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