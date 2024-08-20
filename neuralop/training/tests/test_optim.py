import torch
from torch.nn import Parameter
from torch.testing import assert_close

from ..adamw import AdamW

def test_complex_adamw_momentum():
    param1 = Parameter((0. + 1.0j) * torch.ones((3,3), dtype=torch.cfloat))
    param2 = Parameter((0. + 1.0j) * torch.ones((3,3), dtype=torch.cfloat))

    complex_optimizer = AdamW(params=[param1],
                      betas=(0.5, 0.5))
    orig_optimizer = torch.optim.AdamW(params=[param2],
                                       betas=(0.5, 0.5),)
    

    loss1 = torch.view_as_real((param1 * param1.conj())).sum()
    loss2 = torch.view_as_real((param2 * param2.conj())).sum()
    # grad x^2 = 2x, grads are all 0 + 2j

    loss1.backward()
    complex_optimizer.step()

    # momentum value should be elemwise (2j * -2j * 0.5) = 2 + 0j
    # exp_avg_sq should be empty, meaning it is just momentum * (1-beta2)
    assert_close(complex_optimizer.state[param1]["exp_avg_sq"], (2 + 0j) * torch.ones_like(param1))

    ###### Comparison: wrong optim from torch.optim.Adam
    # values will be a 2-tensor stack [grad.real * zeros_like(grad), grad.imag * ones_like(grad)]
    # grads are 0 + 2j so this will be [zeros, 2 * ones]
    # squaring this, multiplying by 1-beta2=0.5, adding to 0s will be [zeros, 2 * ones]
    # then casting back to complex should return [0 + 2j]
    loss2.backward()
    orig_optimizer.step()

    assert_close(orig_optimizer.state[param2]["exp_avg_sq"], (0 + 2j) * torch.ones_like(param2))
