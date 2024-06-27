from ..data_processors import DefaultDataProcessor, IncrementalDataProcessor
from ..normalizers import UnitGaussianNormalizer
from ..positional_embeddings import GridEmbedding2D

import torch
from torch.testing import assert_close

from neuralop.tests.test_utils import DummyModel

def test_DefaultDataProcessor_pipeline():
    if torch.backends.cuda.is_built():
        device = 'cuda'
    else:
        device='cpu'

    x = torch.randn((1,2,64,64))
    y = torch.randn((1,2,64,64))

    pos_encoder = GridEmbedding2D(grid_boundaries=[[0,1],[0,1]])
    normalizer = UnitGaussianNormalizer(mean=torch.zeros((1,2,1,1)),
                                        std=torch.ones((1,2,1,1)),
                                        eps=1e-5)

    pipeline = DefaultDataProcessor(in_normalizer=normalizer,
                           out_normalizer=normalizer,
                           positional_encoding=pos_encoder)
    
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
    assert wrapped_model.model.training
    wrapped_model.eval()
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
        positional_encoding=None,
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