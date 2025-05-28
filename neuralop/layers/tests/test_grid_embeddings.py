import random

import torch
from torch.testing import assert_close
import pytest

from ..embeddings import GridEmbedding2D, GridEmbeddingND

# Testing grid-based pos encoding: choose a random grid 
# point and assert the proper encoding is applied there

def test_GridEmbedding2D():
    grid_boundaries = [[0,1], [0,1]]
    pos_embed = GridEmbedding2D(in_channels=1,
                                grid_boundaries=grid_boundaries)

    input_res = (20,20)
    x = torch.randn(1,1,*input_res)
    x = pos_embed(x)

    index = [random.randint(0, res-1) for res in input_res]
    true_coords = x[0,1:,index[0], index[1]].squeeze() # grab pos encoding channels at coord index
    expected_coords = torch.tensor([i/j for i,j in zip(index,input_res)])
    assert_close(true_coords, expected_coords)

@pytest.mark.parametrize('dim', [1,2,3,4])
def test_GridEmbeddingND(dim):
    grid_boundaries = [[0,1]] * dim
    pos_embed = GridEmbeddingND(in_channels=1,
                                dim=dim,
                            grid_boundaries=grid_boundaries)

    input_res = [20] * dim
    x = torch.randn(1,1,*input_res)
    x = pos_embed(x)

    index = [random.randint(0, res-1) for res in input_res]
    # grab pos encoding channels at coord index
    pos_channels = x[0,1:,...]
    indices = [slice(None), *index]
    true_coords = pos_channels[indices]
    expected_coords = torch.tensor([i/j for i,j in zip(index,input_res)])
    assert_close(true_coords, expected_coords)