import torch
from ..data_losses import LpLoss, H1Loss, MSELoss
from ..meta_losses import ShakeShakeLoss

from torch.testing import assert_close
from flaky import flaky

@flaky(max_runs=5, min_passes=3)
def test_shakeshake_average():
    n_trials = 50

    lploss = LpLoss(d=2, p=2, L=1.0)
    h1loss = H1Loss(d=2)

    x, y = torch.randn(2, 10, 2), torch.randn(2, 10, 2)
    
    # Compute H1 and Lp separately
    lp_result = lploss(x,y)
    h1_result = h1loss(x,y)

    # assert that over many trials, the average result is the weighted average provided
    shake_shake = ShakeShakeLoss(losses=[h1loss, lploss])
    result_list = []

    for _ in range(n_trials):
        result_list.append(shake_shake(x,y))
    
    avg_result = torch.mean(torch.stack(result_list), dim=0)

    results_averaged = (lp_result + h1_result) / 2

    assert_close(avg_result, results_averaged, atol=1/n_trials, rtol=1e-3)