import torch
from torch.nn import Parameter
from torch.testing import assert_close

from ..adamw import AdamW

def test_complex_adamw_momentum():
    param = Parameter((0. + 1.0j) * torch.ones((3,3), dtype=torch.cfloat))

    optimizer = AdamW(params=[param],
                      betas=(0.5, 0.5),
                      track_momentum=True)

    loss = torch.view_as_real((param * param.conj())).sum()
    # grad x^2 = 2x, grads are all 0 + 2j

    loss.backward()
    optimizer.step()

    # momentum value should be elemwise (2j * -2j * 0.5) = 2 + 0j
    assert_close(optimizer.state[param]["momentum_val"], (2 + 0j) * torch.ones_like(param))
