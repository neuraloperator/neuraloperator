import pytest
import torch
from ..fno_block import FNOBlocks

def test_FNOBlock_output_scaling_factor():
    """Test FNOBlocks with upsampled or downsampled outputs
    """
    max_n_modes = [8, 8, 8, 8]
    n_modes = [4, 4, 4, 4]
    
    size = [10]*4
    mlp_dropout=0
    mlp_expansion=0.5
    mlp_skip='linear'
    for dim in [1, 2, 3, 4]:
        block = FNOBlocks(
            3, 4, max_n_modes[:dim], max_n_modes=max_n_modes[:dim], n_layers=1)
        
        assert block.convs.n_modes[:-1] == max_n_modes[:dim-1]
        assert block.convs.n_modes[-1] == max_n_modes[dim-1]//2 + 1

        block.n_modes = n_modes[:dim]
        assert block.convs.n_modes[:-1] == n_modes[:dim-1]
        assert block.convs.n_modes[-1] == n_modes[dim-1]//2 + 1

        block.n_modes = max_n_modes[:dim]
        assert block.convs.n_modes[:-1] == max_n_modes[:dim-1]
        assert block.convs.n_modes[-1] == max_n_modes[dim-1]//2 + 1

        # Downsample outputs
        block = FNOBlocks(
            3, 4, n_modes[:dim], n_layers=1, output_scaling_factor=0.5, 
            use_mlp=True, mlp_dropout=mlp_dropout, mlp_expansion=mlp_expansion, mlp_skip=mlp_skip)

        x = torch.randn(2, 3, *size[:dim])
        res = block(x)
        assert(list(res.shape[2:]) == [m//2 for m in size[:dim]])
        
        # Upsample outputs
        block = FNOBlocks(
            3, 4, n_modes[:dim], n_layers=1, output_scaling_factor=2,
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

@pytest.mark.parametrize('n_dim', 
                         [1,2,3])
def test_FNOBlock_complex_data(n_dim):
    """Test FNO layers with complex input data
    """
    modes = (8, 8, 8)
    size = [10]*3
    mlp_dropout=0
    mlp_expansion=0.5
    mlp_skip='linear'
    # Instantiate a complex-valued FNO block
    block = FNOBlocks(
        3, 4, modes[:n_dim], n_layers=1, use_mlp=True,
        mlp_dropout=mlp_dropout, mlp_expansion=mlp_expansion, mlp_skip=mlp_skip, complex_data=True)

    x = torch.randn(2, 3, *size[:n_dim], dtype=torch.cfloat)
    res = block(x)

    assert(list(res.shape[2:]) == size[:n_dim])
    
    
def test_FNOBlock_max_n_modes_setter():
    """Test FNOBlocks with updating max_modes
    """
    modes = (4, 4, 4)
    max_n_modes = (6, 6, 6)
    updated_max_n_modes = (8, 8, 8)
    for dim in [1, 2, 3]:
        # Downsample outputs
        block = FNOBlocks(
            3, 3, modes[:dim], max_n_modes=max_n_modes[:dim], n_layers=1)
    
        assert block.convs.max_n_modes == list(max_n_modes[:dim]) # check defaults
        
        block.convs.max_n_modes = updated_max_n_modes[:dim]
        
        assert block.convs.max_n_modes == list(updated_max_n_modes[:dim]) # check updated value