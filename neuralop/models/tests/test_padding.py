import torch
from ..padding import DomainPadding
import pytest

@pytest.mark.parametrize('mode', ['one-sided', 'symmetric'])
def test_DomainPadding(mode):
    data = torch.randn((2, 3, 10, 10))
    padder = DomainPadding(0.2, mode)
    padded = padder.pad(data)
    unpadded = padder.unpad(padded)
    assert unpadded.shape == data.shape

