import pytest
import torch
from ..fno_block import FNOBlocks

def test_FNOBlock_resolution_scaling_factor():
    """Test FNOBlocks with upsampled or downsampled outputs
    """
    max_n_modes = [8, 8, 8, 8]
    n_modes = [4, 4, 4, 4]
    
    size = [10]*4
    channel_mlp_dropout=0
    channel_mlp_expansion=0.5
    channel_mlp_skip='linear'
    for dim in [1, 2, 3, 4]:
        block = FNOBlocks(
            3, 4, max_n_modes[:dim], max_n_modes=max_n_modes[:dim], n_layers=1, channel_mlp_skip=channel_mlp_skip)
        
        assert block.convs[0].n_modes[:-1] == max_n_modes[:dim-1]
        assert block.convs[0].n_modes[-1] == max_n_modes[dim-1]//2 + 1

        block.n_modes = n_modes[:dim]
        assert block.convs[0].n_modes[:-1] == n_modes[:dim-1]
        assert block.convs[0].n_modes[-1] == n_modes[dim-1]//2 + 1

        block.n_modes = max_n_modes[:dim]
        assert block.convs[0].n_modes[:-1] == max_n_modes[:dim-1]
        assert block.convs[0].n_modes[-1] == max_n_modes[dim-1]//2 + 1

        # Downsample outputs
        block = FNOBlocks(
            3, 4, n_modes[:dim], n_layers=1, resolution_scaling_factor=0.5, 
            use_mlp=True, channel_mlp_dropout=channel_mlp_dropout, 
            channel_mlp_expansion=channel_mlp_expansion, channel_mlp_skip=channel_mlp_skip)

        x = torch.randn(2, 3, *size[:dim])
        res = block(x)
        assert(list(res.shape[2:]) == [m//2 for m in size[:dim]])
        
        # Upsample outputs
        block = FNOBlocks(
            3, 4, n_modes[:dim], n_layers=1, resolution_scaling_factor=2, 
            channel_mlp_dropout=channel_mlp_dropout, channel_mlp_expansion=channel_mlp_expansion, 
            channel_mlp_skip=channel_mlp_skip)

        x = torch.randn(2, 3, *size[:dim])
        res = block(x)
        assert res.shape[1] == 4 # Check out channels
        assert(list(res.shape[2:]) == [m*2 for m in size[:dim]])


@pytest.mark.parametrize('n_dim', [1,2,3,4])

@pytest.mark.parametrize('norm', 
                         ['instance_norm', 'ada_in', 'group_norm', 'batch_norm'])
def test_FNOBlock_norm(norm, n_dim):
    """Test SpectralConv with upsampled or downsampled outputs
    """
    modes = (8, 8, 8)
    size = [10]*3
    channel_mlp_dropout=0
    channel_mlp_expansion=0.5
    channel_mlp_skip='linear'
    ada_in_features = 4
    block = FNOBlocks(
        3, 4, modes[:n_dim], n_layers=1, norm=norm, ada_in_features=ada_in_features,
        channel_mlp_dropout=channel_mlp_dropout, channel_mlp_expansion=channel_mlp_expansion, 
        channel_mlp_skip=channel_mlp_skip)

    if norm == 'ada_in':
        embedding = torch.randn(ada_in_features)
        block.set_ada_in_embeddings(embedding)

    x = torch.randn(2, 3, *size[:n_dim])
    res = block(x)
    assert(list(res.shape[2:]) == size[:n_dim])

@pytest.mark.parametrize('n_dim', 
                         [1,2,3])
def test_FNOBlock_complex_data(n_dim):
    """Test FNO layers with complex input data
    """
    modes = (8, 8, 8)
    size = [10]*3
    channel_mlp_dropout=0
    mlp_expansion=0.5
    channel_mlp_skip='linear'
    # Instantiate a complex-valued FNO block
    block = FNOBlocks(
        3, 4, modes[:n_dim], n_layers=1,
        channel_mlp_dropout=channel_mlp_dropout, channel_mlp_expansion=mlp_expansion, 
        channel_mlp_skip=channel_mlp_skip, complex_data=True)

    x = torch.randn(2, 3, *size[:n_dim], dtype=torch.cfloat)
    res = block(x)

    assert(list(res.shape[2:]) == size[:n_dim])