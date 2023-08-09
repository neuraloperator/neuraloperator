import torch
from ..padding import DomainPadding
import pytest

@pytest.mark.parametrize('mode', ['one-sided', 'symmetric'])
@pytest.mark.parametrize('padding_dim', [0.2, [0.1, 0.2]])
def test_DomainPadding_2d(mode, padding_dim):
    if isinstance(padding_dim, float):
        out_size =  {'one-sided': [12, 12], 'symmetric': [14, 14]}
    else:
        out_size =  {'one-sided': [11, 12], 'symmetric': [12, 14]} 

    data = torch.randn((2, 3, 10, 10))
    padder = DomainPadding(padding_dim, mode)
    padded = padder.pad(data)

    target_shape = list(padded.shape)
    for ctr in range(1,3):
        target_shape[-ctr] = out_size[mode][-ctr]
    assert list(padded.shape) == target_shape

    unpadded = padder.unpad(padded)
    assert unpadded.shape == data.shape


@pytest.mark.parametrize('mode', ['one-sided', 'symmetric'])
@pytest.mark.parametrize('padding_dim', [0.2, [0.1, 0, 0.2]])
def test_DomainPadding_3d(mode, padding_dim):
    if isinstance(padding_dim, float):
        out_size =  {'one-sided': [12, 12, 12], 'symmetric': [14, 14, 14]}
    else:
        out_size =  {'one-sided': [11, 10, 12], 'symmetric': [12, 10, 14]} 

    data = torch.randn((2, 3, 10, 10, 10))
    padder = DomainPadding(padding_dim, mode)
    padded = padder.pad(data)

    target_shape = list(padded.shape)
    for ctr in range(1,4):
        target_shape[-ctr] = out_size[mode][-ctr]
    assert list(padded.shape) == target_shape

    unpadded = padder.unpad(padded)
    assert unpadded.shape == data.shape


