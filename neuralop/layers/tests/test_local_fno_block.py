import pytest
import torch
from ..local_fno_block import LocalFNOBlocks

def test_LocalFNOBlock_output_scaling_factor():
    """Test LocalFNOBlocks with upsampled or downsampled outputs
    """
    max_n_modes = [8, 8, 8]
    n_modes = [4, 4, 4]
    
    size = [10]*3
    mlp_dropout=0
    mlp_expansion=0.5
    mlp_skip='linear'
    for dim in [1, 2, 3]:
        block = LocalFNOBlocks(
            3, 4, max_n_modes[:dim], max_n_modes=max_n_modes[:dim], n_layers=1, diff_layers=[True])
        
        assert block.convs.n_modes[:-1] == max_n_modes[:dim-1]
        assert block.convs.n_modes[-1] == max_n_modes[dim-1]//2 + 1

        block.n_modes = n_modes[:dim]
        assert block.convs.n_modes[:-1] == n_modes[:dim-1]
        assert block.convs.n_modes[-1] == n_modes[dim-1]//2 + 1

        block.n_modes = max_n_modes[:dim]
        assert block.convs.n_modes[:-1] == max_n_modes[:dim-1]
        assert block.convs.n_modes[-1] == max_n_modes[dim-1]//2 + 1

        # Downsample outputs
        block = LocalFNOBlocks(
            3, 4, n_modes[:dim], n_layers=1, diff_layers=[True], output_scaling_factor=0.5, 
            use_mlp=True, mlp_dropout=mlp_dropout, mlp_expansion=mlp_expansion, mlp_skip=mlp_skip)

        x = torch.randn(2, 3, *size[:dim])
        res = block(x)
        assert(list(res.shape[2:]) == [m//2 for m in size[:dim]])
        
        # Upsample outputs
        block = LocalFNOBlocks(
            3, 4, n_modes[:dim], n_layers=1, diff_layers=[True], output_scaling_factor=2,
            use_mlp=True, mlp_dropout=mlp_dropout, mlp_expansion=mlp_expansion, mlp_skip=mlp_skip)

        x = torch.randn(2, 3, *size[:dim])
        res = block(x)
        assert res.shape[1] == 4 # Check out channels
        assert(list(res.shape[2:]) == [m*2 for m in size[:dim]])