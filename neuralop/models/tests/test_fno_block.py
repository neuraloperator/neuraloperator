import torch
from ..fno_block import FNOBlocks

def test_FactorizedSpectralConv_output_scaling_factor():
    """Test FactorizedSpectralConv with upsampled or downsampled outputs
    """
    modes = (8, 8, 8)
    incremental_modes = (4, 4, 4)
    size = [10]*3
    mlp = dict(dropout=0, expansion=0.5)
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
            3, 4, modes[:dim], n_layers=1, output_scaling_factor=0.5, use_mlp=True, mlp=mlp, mlp_skip=mlp_skip)

        x = torch.randn(2, 3, *size[:dim])
        res = block(x)
        assert(list(res.shape[2:]) == [m//2 for m in size[:dim]])
        
        # Upsample outputs
        block = FNOBlocks(
            3, 4, modes[:dim], n_layers=1, output_scaling_factor=2, use_mlp=True, mlp=mlp, mlp_skip=mlp_skip)

        x = torch.randn(2, 3, *size[:dim])
        res = block(x)
        assert res.shape[1] == 4 # Check out channels
        assert(list(res.shape[2:]) == [m*2 for m in size[:dim]])

