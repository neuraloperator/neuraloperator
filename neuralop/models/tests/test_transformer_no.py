import time
import torch
import pytest
import math
from random import shuffle
from torch.testing import assert_close

from ..transformer_no import TransformerNO
from neuralop.layers.embeddings import regular_grid_nd


@pytest.mark.parametrize(
    "input_shape", [(32, 2048, 1), (32, 64, 64, 3)],   # a 1D case and a 2D case,
)
@pytest.mark.parametrize(
    "regular_grid", [True, False],
)
def test_TransformerNO(input_shape, regular_grid):
    if torch.has_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu:0')

    if len(input_shape) == 3:
        n_dim = 1
    else:
        n_dim = 2
    batch_size = input_shape[0]
    data_channels = input_shape[-1]
    model = TransformerNO(n_dim=n_dim,
                          in_channels=data_channels,
                          out_channels=1,    # single channel output
                          encoder_hidden_channels=32,
                          decoder_hidden_channels=32,
                          encoder_num_heads=4,
                          decoder_num_heads=4,
                          encoder_head_n_channels=32,
                          decoder_head_n_channels=32,
                          encoder_n_layers=3,
                          query_basis='siren')
    model.to(device)

    in_data = torch.randn(input_shape).to(device)
    in_data = in_data.view(batch_size, -1, in_data.shape[-1])   # flatten the spatial dimensions

    if regular_grid:
        # apply regular grid-based pos embedding
        grid_shape = input_shape[1:-1]
        grids =  regular_grid_nd(resolutions=grid_shape, grid_boundaries=[[0,1]] * len(grid_shape))
        if len(grid_shape) == 1:
            pos_reg = grids[0].unsqueeze(-1)
        else:
            pos_reg = torch.cat([x.unsqueeze(-1) for x in grids], dim=-1)
        pos = pos_reg.view(1, -1, pos_reg.shape[-1]).to(device)
    else:
        pos = torch.randn(1, in_data.shape[1], n_dim).to(device)

    # Test forward pass
    out = model(in_data, pos)

    # Check output size
    output_size = math.prod(input_shape[1:-1])
    assert list(out.shape) == [input_shape[0], output_size, 1]

    # Check backward pass
    loss = out.sum()
    loss.backward()

    # make sure the ordering of the points does not change output
    mesh_pt_indices = in_data.shape[1]
    indices = list(range(mesh_pt_indices))
    shuffle(indices)
    in_data_shuffled = in_data[:, indices, ...]
    pos_shuffled = pos[:, indices, ...]

    with torch.no_grad():
        out_shuffled = model(in_data_shuffled, pos_shuffled)
        out_unshuffled = model(in_data, pos)
        assert_close(out_unshuffled[:, indices, ...], out_shuffled)

    n_unused_params = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            n_unused_params += 1
            print(name)
    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"
