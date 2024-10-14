import torch
from torch.nn import Parameter
from torch.testing import assert_close
import pytest

from ..adamw import AdamW

@pytest.mark.parametrize('adam_optimizer_cls', [AdamW])
def test_correct_complex_adam_momentum(adam_optimizer_cls):
    # param = x * 2j
    x = torch.randn((3,3), dtype=torch.float64)
    param = Parameter(((0. + 1.0j) * x).to(torch.cfloat))

    optimizer = adam_optimizer_cls(params=[param],
                      betas=(0.5, 0.5))

    loss = torch.view_as_real((param * param.conj())).sum()
    # grad x^2 = 2x, grads are all 0 + 2j * x

    loss.backward()
    optimizer.step()

    # momentum value should be elemwise (2jx * -2jx * (1 - 0.5)) = 4x**2 * 0.5 = 2x**2
    # exp_avg_sq should be empty, meaning it is just momentum * (1-beta2)
    momentum = optimizer.state[param]["exp_avg_sq"]
    assert_close(momentum, (2 * x**2).to(torch.cfloat))
