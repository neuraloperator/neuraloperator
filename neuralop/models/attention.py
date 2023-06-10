from .fno_block import FNOBlocks
import torch.nn as nn
from .spectral_convolution import FactorizedSpectralConv2d, SpectralConvKernel2d
import torch.nn.functional as F
import torch
class TnoBlock2d(nn.Module):
    def __init__(self,in_channels, out_channels, n_modes,
                 output_scaling_factor=None,
                 incremental_n_modes=None,
                 use_mlp=False, mlp=None, mlp_dropout=0, mlp_expansion=0.5,
                 non_linearity=F.gelu,
                 norm=None, preactivation=False,
                 fno_skip='linear',
                 mlp_skip='soft-gating',
                 separable=False,
                 factorization=None,
                 rank=1.0,
                 SpectralConv=SpectralConvKernel2d,
                 joint_factorization=False, 
                 fixed_rank_modes=False,
                 implementation='factorized',
                 decomposition_kwargs=dict(),
                 fft_norm='forward', output_shape = None, normalizer = None):
        
        super().__init__()
        self.K = FNOBlocks(in_channels= in_channels, out_channels= out_channels, n_modes= n_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor = output_scaling_factor,non_linearity=non_linearity,\
                                            norm=norm, preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip,mlp_dropout=0, mlp_expansion=0.5,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                            SpectralConv= SpectralConv,n_layers=1)

        self.Q = FNOBlocks(in_channels= in_channels, out_channels= out_channels, n_modes= n_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor = output_scaling_factor,non_linearity=non_linearity,\
                                            norm=norm, preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip, mlp_dropout=0, mlp_expansion=0.5,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                            SpectralConv= SpectralConv, n_layers=1)

        self.V = FNOBlocks(in_channels= in_channels, out_channels= out_channels, n_modes= n_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor = output_scaling_factor,non_linearity=non_linearity,\
                                            norm=norm, preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip, mlp_dropout=0, mlp_expansion=0.5,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                            SpectralConv= SpectralConv,n_layers=1)
        self.nomalizer = normalizer
        if normalizer is not None:
            self.non_lin = non_linearity
            self.normalize_layer = torch.nn.InstanceNorm2d(int(out_channels),affine=False)

        
    def forward(self,x, output_shape = None):
        k = self.K(x, output_shape = output_shape)
        q = self.Q(x, output_shape = output_shape)
        v = self.V(x, output_shape = output_shape)
        
        batch, codim = k.shape[0], k.shape[1]
        
        k = k.view(batch, codim, -1).transpose(-1,-2)
        q = q.view(batch, codim, -1)
        
        

        # normalize dot product implemented for 2D data(latitute and longitude) 

        dprod =  torch.matmul(q, k)/k.shape[-1]

        dprod = F.softmax(dprod, dim = -1)
        
        #print(attention.shape, v.shape)

        # value
        value_x, value_y = v.shape[-2], v.shape[-1]
        
        v = v.view(batch, codim, -1)

        output =  torch.matmul(dprod, v)
        
        output = output.view(batch, codim, value_x, value_y)
        
        if self.nomalizer is not None:
            output = self.non_lin(self.normalize_layer(output))

        return output

