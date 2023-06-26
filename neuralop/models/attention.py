from .fno_block import FNOBlocks
import torch.nn as nn
from .spectral_convolution import FactorizedSpectralConv
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from .fino import SpectralConvKernel2d
import torch

class TnoBlock2d(nn.Module):
    def __init__(self,in_channels, n_modes,
                 n_head = 1,
                 token_codim = 1,
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
                 fft_norm='forward', normalizer = 'layer_norm', **kwarg):
        
        super().__init__()
        self.token_codim = token_codim
        self.n_head = n_head
        self.output_scaling_factor = output_scaling_factor
        
        mixer_modes = [i//self.n_head for i in n_modes]
        self.K = FNOBlocks(in_channels= self.token_codim, out_channels= self.n_head * self.token_codim, n_modes= mixer_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor = 1/n_head,non_linearity=non_linearity,\
                                            norm=norm, preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip,mlp_dropout=0, mlp_expansion=0.5,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                            SpectralConv= SpectralConv,n_layers=1)

        self.Q = FNOBlocks(in_channels= self.token_codim, out_channels= self.n_head * self.token_codim, n_modes= mixer_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor = 1/n_head,non_linearity=non_linearity,\
                                            norm=norm, preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip, mlp_dropout=0, mlp_expansion=0.5,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                            SpectralConv= SpectralConv, n_layers=1)

        self.V = FNOBlocks(in_channels= self.token_codim, out_channels= self.n_head * self.token_codim, n_modes= mixer_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor = 1/n_head,non_linearity=non_linearity,\
                                            norm=norm, preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip, mlp_dropout=0, mlp_expansion=0.5,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                            SpectralConv= SpectralConv,n_layers=1)
        self.nomalizer = normalizer
        
        if n_head != 1:
            self.mixer = FNOBlocks(in_channels= self.n_head * self.token_codim, out_channels= self.token_codim, n_modes= mixer_modes,\
                                                use_mlp=use_mlp, mlp=mlp, output_scaling_factor = n_head,non_linearity=non_linearity,\
                                                norm=norm, preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip, mlp_dropout=0, mlp_expansion=0.5,\
                                                incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
                                                fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                                factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                                SpectralConv= SpectralConv,n_layers=1) 
        
        self.end_block = FNOBlocks(in_channels= in_channels, out_channels= in_channels, n_modes= n_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor = self.output_scaling_factor,non_linearity=non_linearity,\
                                            norm=norm, preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip, mlp_dropout=0, mlp_expansion=0.5,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                            SpectralConv= SpectralConv,n_layers=1)
        if normalizer is not None:
            self.non_lin = non_linearity
            self.normalize_layer = torch.nn.InstanceNorm2d(int(in_channels),affine=False)

        
    def forward(self, x, output_shape = None):
        batch, n_token , in_res_x, in_res_y = x.shape[0], x.shape[1]//self.token_codim, x.shape[-2], x.shape[-1]
        
        
        assert x.shape[1]%self.token_codim == 0
        
        x = rearrange(x, 'b (t d) h w -> b t d h w', d=self.token_codim)
        x = rearrange(x, 'b t d h w -> (b t) d h w')
        #print(x.shape)
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)
        
        res_x, res_y = k.shape[-2], k.shape[-1]
        value_res_x, value_res_y = v.shape[-2], v.shape[-1]
        
        k = rearrange(k, '(b t) (a d) h w -> b a t (d h w)', b=batch, a=self.n_head )
        q = rearrange(q, '(b t) (a d) h w -> b a t (d h w)', b=batch, a=self.n_head )
        v = rearrange(v, '(b t) (a d) h w -> b a t (d h w)', b=batch, a=self.n_head )


        dprod = torch.matmul(q, k.transpose(-1, -2))/k.shape[-1]

        dprod = F.softmax(dprod, dim = -1)
                
        output =  torch.matmul(dprod, v)
        
        output = rearrange(output, 'b a t (d h w) -> b t a d h w', d=self.token_codim, h=value_res_x, w=value_res_y)
        
        if self.n_head != 1:
            output = rearrange(output, 'b t a d h w -> (b t) (a d) h w')
            output = self.mixer(output, output_shape = (in_res_x, in_res_y))
            output = rearrange(output, '(b t) d h w -> b (t d) h w', b=batch)
        else:
            output = output = rearrange(output, 'b t a d h w -> b (t a d) h w')
        
        
        if self.nomalizer is not None:
            output = self.non_lin(self.normalize_layer(output))


        
        output =  self.end_block(output, output_shape = output_shape)
        
        return output