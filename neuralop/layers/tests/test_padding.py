import torch
from ..padding import DomainPadding
import pytest

@pytest.mark.parametrize('mode', ['one-sided', 'symmetric'])
def test_DomainPadding(mode):
    out_size = {'one-sided': 12, 'symmetric': 14}
    data = torch.randn((2, 3, 10, 10))
    padder = DomainPadding(0.2, mode)
    padded = padder.pad(data)

    target_shape = list(padded.shape)
    target_shape[-1] = target_shape[-2] = out_size[mode]
    assert list(padded.shape) == target_shape

    unpadded = padder.unpad(padded)
    assert unpadded.shape == data.shape

