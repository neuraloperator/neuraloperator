import pytest
import torch
from ..fno_block import FNOBlocks

def test_FNOBlock_output_scaling_factor():
    """Test FNOBlocks with upsampled or downsampled outputs
    """
    modes = (8, 8, 8)
    incremental_modes = (4, 4, 4)
    size = [10]*3
    mlp_dropout=0
    mlp_expansion=0.5
    mlp_skip='linear'
    for dim in [1, 2, 3]:
        block = FNOBlocks(
            3, 4, modes[:dim], n_layers=1)
        
        assert block.convs.n_modes == modes[:dim]

        block.incremental_n_modes = incremental_modes[:dim]
        assert block.convs.incremental_n_modes == incremental_modes[:dim]

        block.incremental_n_modes = modes[:dim]
        assert block.convs.incremental_n_modes == modes[:dim]

        # Downsample outputs
        block = FNOBlocks(
            3, 4, modes[:dim], n_layers=1, output_scaling_factor=0.5, 
            use_mlp=True, mlp_dropout=mlp_dropout, mlp_expansion=mlp_expansion, mlp_skip=mlp_skip)

        x = torch.randn(2, 3, *size[:dim])
        res = block(x)
        assert(list(res.shape[2:]) == [m//2 for m in size[:dim]])
        
        # Upsample outputs
        block = FNOBlocks(
            3, 4, modes[:dim], n_layers=1, output_scaling_factor=2,
            use_mlp=True, mlp_dropout=mlp_dropout, mlp_expansion=mlp_expansion, mlp_skip=mlp_skip)

        x = torch.randn(2, 3, *size[:dim])
        res = block(x)
        assert res.shape[1] == 4 # Check out channels
        assert(list(res.shape[2:]) == [m*2 for m in size[:dim]])


@pytest.mark.parametrize('norm', 
                         ['instance_norm', 'ada_in', 'group_norm'])
def test_FNOBlock_norm(norm):
    """Test SpectralConv with upsampled or downsampled outputs
    """
    modes = (8, 8, 8)
    size = [10]*3
    mlp_dropout=0
    mlp_expansion=0.5
    mlp_skip='linear'
    dim = 2
    ada_in_features = 4
    block = FNOBlocks(
        3, 4, modes[:dim], n_layers=1, use_mlp=True, norm=norm, ada_in_features=ada_in_features,
        mlp_dropout=mlp_dropout, mlp_expansion=mlp_expansion, mlp_skip=mlp_skip)

    if norm == 'ada_in':
        embedding = torch.randn(ada_in_features)
        block.set_ada_in_embeddings(embedding)

    x = torch.randn(2, 3, *size[:dim])
    res = block(x)
    assert(list(res.shape[2:]) == size[:dim])