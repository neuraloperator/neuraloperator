import torch
from torch.nn import Parameter
from torch.testing import assert_close
import pytest

from ..adamw import AdamW

@pytest.mark.parametrize('adam_optimizer_cls', [AdamW])
def test_correct_complex_adam_momentum(adam_optimizer_cls):
    param = Parameter((0. + 1.0j) * torch.ones((3,3), dtype=torch.cfloat))

    optimizer = adam_optimizer_cls(params=[param],
                      betas=(0.5, 0.5))

    loss = torch.view_as_real((param * param.conj())).sum()
    # grad x^2 = 2x, grads are all 0 + 2j

    loss.backward()
    optimizer.step()

    # momentum value should be elemwise (2j * -2j * 0.5) = 2 + 0j
    # exp_avg_sq should be empty, meaning it is just momentum * (1-beta2)
    assert_close(optimizer.state[param]["exp_avg_sq"], (2 + 0j) * torch.ones_like(param))

