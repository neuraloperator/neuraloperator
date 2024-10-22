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
@pytest.mark.parametrize('stitching', [True, False])
@pytest.mark.parametrize('evaluation', [True, False])

def test_full_mgp2d(levels, padding_fraction, stitching, evaluation):

    model = DummyModel(16)
    patcher = MultigridPatching2D(model=model,
                                  levels=levels,
                                  padding_fraction=padding_fraction,
                                  stitching=stitching, # cpu-only, single process
                                  use_distributed=False)
    
    input_shape = (batch_size, channels, side_len, side_len)
    x = torch.randn(*input_shape)
    y = torch.randn(*input_shape)

    patched_x, patched_y = patcher.patch(x,y)
    n_patches = 2 ** levels
    padding = int(round(side_len * padding_fraction))
    patched_padded_side_len = int((side_len // n_patches) + (2 * padding))

    assert patched_x.shape ==\
          ((n_patches ** 2) * batch_size, channels + levels, patched_padded_side_len, patched_padded_side_len)
    
    # mimic output after scattering x to model parallel region
    patched_out_shape = (patched_x.shape[0], 1, *patched_x.shape[2:])
    patched_out = torch.randn(patched_out_shape)
    
    # if padding is not applied, return without stitching
    # otherwise unpad and stitch

    if stitching or evaluation:
        unpatch_shape = input_shape
    else:
        unpadded_patch_size = int(side_len // n_patches)
        unpatch_shape = (patched_x.shape[0], channels, unpadded_patch_size, unpadded_patch_size)

    # test stitching here in cases where padding is applied
    unpatched_x, unpatched_y = patcher.unpatch(patched_out, patched_y, evaluation=evaluation)
        
    assert unpatched_x.shape == unpatch_shape
    assert unpatched_y.shape == unpatch_shape
