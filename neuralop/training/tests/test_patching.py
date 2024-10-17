import torch
import pytest

from ..patching import MultigridPatching2D, make_patches
from neuralop.tests.test_utils import DummyModel

# Input shape params
batch_size = 16
channels = 1
side_len = 128

@pytest.mark.parametrize('levels', [1, 2, 3])
@pytest.mark.parametrize('padding_fraction', [0, 0.1, 0.2])
def test_make_patches(levels, padding_fraction):
    x = torch.randn(batch_size, channels, side_len, side_len)
    n_patches = 2 ** levels
    
    padding = int(round(side_len * padding_fraction))
    patched_x = make_patches(x, n_patches, padding)

    patched_side_len = int((side_len // n_patches) + (2 * padding))
    assert patched_x.shape == ((n_patches ** 2) * batch_size, channels, patched_side_len, patched_side_len)

@pytest.mark.parametrize('levels', [1, 2, 3])
@pytest.mark.parametrize('padding_fraction', [0, 0.1, 0.2])
def test_full_mgp2d(levels, padding_fraction):

    model = DummyModel(16)
    patcher = MultigridPatching2D(model=model,
                                  levels=levels,
                                  padding_fraction=padding_fraction,
                                  stitching=False, # cpu-only, single process
                                  use_distributed=False)
    
    x = torch.randn(batch_size, channels, side_len, side_len)
    y = torch.randn(batch_size, channels, side_len, side_len)

    patched_x, patched_y = patcher.patch(x,y)
    n_patches = 2 ** levels
    padding = int(round(side_len * padding_fraction))
    patched_padded_side_len = int((side_len // n_patches) + (2 * padding))
    unpatch_side_len = int(side_len - (2 * padding))

    assert patched_x.shape ==\
          ((n_patches ** 2) * batch_size, channels + levels, patched_padded_side_len, patched_padded_side_len)
    
    unpatched_x, unpatched_y = patcher.unpatch(x,y)
        
    assert unpatched_x.shape == (batch_size, channels, unpatch_side_len, unpatch_side_len)
