import torch
from ..padding import DomainPadding
import pytest

def test_DomainPadding():
    data = torch.randn((2, 3, 10, 10))
    padder = DomainPadding(0.2)
    padded = padder.pad(data)

    # With 0.2 padding and symmetric mode, output size should be 14
    target_shape = list(padded.shape)
    target_shape[-1] = target_shape[-2] = 14
    assert list(padded.shape) == target_shape

    unpadded = padder.unpad(padded)
    assert unpadded.shape == data.shape

