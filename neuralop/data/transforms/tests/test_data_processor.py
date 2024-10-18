import pytest

import torch
from torch.testing import assert_close

from ..data_processors import DefaultDataProcessor, IncrementalDataProcessor, MGPatchingDataProcessor
from ..normalizers import UnitGaussianNormalizer

from neuralop.tests.test_utils import DummyModel

def test_DefaultDataProcessor_pipeline():
    if torch.backends.cuda.is_built():
        device = 'cuda'
    else:
        device='cpu'

    x = torch.randn((1,2,64,64))
    y = torch.randn((1,2,64,64))

    normalizer = UnitGaussianNormalizer(mean=torch.zeros((1,2,1,1)),
                                        std=torch.ones((1,2,1,1)),
                                        eps=1e-5)

    pipeline = DefaultDataProcessor(in_normalizer=normalizer,
                           out_normalizer=normalizer)
    
    data = {'x':x, 'y':y} # data on cpu at this point

    xform_data = pipeline.preprocess(data)

    # model outputs will be on device by default
    out = torch.randn((1,2,64,64)).to(device) 
    
    _, inv_xform_data = pipeline.postprocess(out, xform_data)

    assert_close(inv_xform_data['y'].cpu(), data['y'])


def test_DefaultDataProcessor_train_eval():
    if torch.backends.cuda.is_built():
        device = 'cuda'
    else:
        device='cpu'

    model = DummyModel(features=10)

    normalizer = UnitGaussianNormalizer(mean=torch.zeros((1,2,1,1)),
                                        std=torch.ones((1,2,1,1)),
                                        eps=1e-5)

    pipeline = DefaultDataProcessor(in_normalizer=normalizer,
                           out_normalizer=normalizer)
    wrapped_model = pipeline.wrap(model).to(device)

    assert wrapped_model.device == device
    
    wrapped_model.train()
    assert wrapped_model.training
    assert wrapped_model.model.training

    wrapped_model.eval()
    assert not wrapped_model.training
    assert not wrapped_model.model.training

# ensure that the data processor incrementally increases the resolution
def test_incremental_resolution():
    if torch.backends.cuda.is_built():
        device = 'cuda'
    else:
        device='cpu'

    x = torch.randn((1,2,16,16)).to(device)
    y = torch.randn((1,2,16,16)).to(device)
    indice_list = [2, 3]
    
    data_transform = IncrementalDataProcessor(
        in_normalizer=None,
        out_normalizer=None,
        device=device,
        subsampling_rates=[2],
        dataset_resolution=16,
        dataset_indices=indice_list,
        epoch_gap=10,
        verbose=True,
    )

    # model outputs will be on device by default
    x_new, y_new = data_transform.regularize_input_res(x, y)
    
    # This is cause x_new.shape = (1, 2, 8, 8)
    # This is cause y_new.shape = (1, 2, 8, 8)
    for i in indice_list:
        assert x_new.shape[i] < x.shape[i]
        assert y_new.shape[i] < y.shape[i]

@pytest.mark.parametrize('levels', [1, 2])
@pytest.mark.parametrize('padding_fraction', [0, 0.1])
def test_full_mgp2d(levels, padding_fraction):
    batch_size = 16
    channels = 1
    side_len = 32

    model = DummyModel(16)
    processor = MGPatchingDataProcessor(model=model,
                                        levels=levels,
                                        padding_fraction=padding_fraction,
                                        stitching=False, # cpu-only, single process
                                        use_distributed=False)
    processor.train()
    
    x = torch.randn(batch_size, channels, side_len, side_len)
    y = torch.randn(batch_size, channels, side_len, side_len)
    sample = {'x': x, 'y': y}
    patched_sample = processor.preprocess(sample)

    n_patches = 2 ** levels
    padding = int(round(side_len * padding_fraction))
    patched_padded_side_len = int((side_len // n_patches) + (2 * padding))
    unpatch_side_len = int(side_len // n_patches)

    assert patched_sample['x'].shape ==\
          ((n_patches ** 2) * batch_size, channels + levels, patched_padded_side_len, patched_padded_side_len)
    
    # mimic output after scattering x to model parallel region
    patched_out_shape = (batch_size, 1, *patched_sample['x'].shape[2:])
    patched_out = torch.randn(patched_out_shape)
    
    unpatched_out, unpatched_y = processor.postprocess(patched_out, patched_sample)
        
    assert unpatched_out.shape == (batch_size, channels, unpatch_side_len, unpatch_side_len)
