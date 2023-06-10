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
                 token_codim = 4,
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
        self.token_codim = token_codim
        self.n_head = int(out_channels/in_channels)
        self.K = FNOBlocks(in_channels= self.token_codim, out_channels= self.token_codim, n_modes= n_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor = output_scaling_factor,non_linearity=non_linearity,\
                                            norm=norm, preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip,mlp_dropout=0, mlp_expansion=0.5,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                            SpectralConv= SpectralConv,n_layers=1)

        self.Q = FNOBlocks(in_channels= self.token_codim, out_channels= self.token_codim, n_modes= n_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor = output_scaling_factor,non_linearity=non_linearity,\
                                            norm=norm, preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip, mlp_dropout=0, mlp_expansion=0.5,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                            SpectralConv= SpectralConv, n_layers=1)

        self.V = FNOBlocks(in_channels= self.token_codim, out_channels= self.token_codim, n_modes= n_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor = output_scaling_factor,non_linearity=non_linearity,\
                                            norm=norm, preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip, mlp_dropout=0, mlp_expansion=0.5,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                            SpectralConv= SpectralConv,n_layers=1)
        self.nomalizer = normalizer
        
        self.end_block = FNOBlocks(in_channels= in_channels, out_channels= out_channels, n_modes= n_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor = None ,non_linearity=non_linearity,\
                                            norm=norm, preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip, mlp_dropout=0, mlp_expansion=0.5,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                            SpectralConv= SpectralConv,n_layers=1)
        if normalizer is not None:
            self.non_lin = non_linearity
            self.normalize_layer = torch.nn.InstanceNorm2d(int(out_channels),affine=False)

        
    def forward(self, x, output_shape = None):
        batch, n_token , res_x, res_y = x.shape[0], x.shape[1]//self.token_codim, x.shape[-2], x.shape[-1]
        
        
        assert x.shape[1]%self.token_codim == 0
        
        x = x.view(-1, self.token_codim, res_x, res_y)
        #print(x.shape)
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x, output_shape = output_shape)
        
        #print("vales and keys 0", v.shape, k.shape)
        
        res_x, res_y = k.shape[-2], k.shape[-1]
        value_res_x, value_res_y = v.shape[-2], v.shape[-1]
        
        k = k.view(batch, n_token, self.token_codim * self.n_head, res_x, res_y)
        q = q.view(batch, n_token, self.token_codim * self.n_head, res_x, res_y)
        v = v.view(batch, n_token, self.token_codim * self.n_head, value_res_x, value_res_y)
        
        #print("vales and keys 1", v.shape, k.shape)
        
        
        k = k.view(batch, n_token, self.n_head, self.token_codim, res_x, res_y).permute(0, 2,1,3,4,5)
        q = q.view(batch, n_token, self.n_head, self.token_codim, res_x, res_y).permute(0, 2,1,3,4,5)
        v = v.view(batch, n_token, self.n_head, self.token_codim, value_res_x, value_res_y).permute(0, 2,1,3,4,5)
        
        #print("vales and keys 2", v.shape, k.shape)
        
        
        k = k.view(batch, self.n_head, n_token, -1).transpose(-1,-2)
        q = q.view(batch,  self.n_head, n_token, -1)
        
        

        # normalize dot product implemented for 2D data(latitute and longitude) 

        dprod =  torch.matmul(q, k)/k.shape[-1]

        dprod = F.softmax(dprod, dim = -1)
        
        ##print(attention.shape, v.shape)

        # value
        
        
        v = v.view(batch, self.n_head, n_token, -1)

        output =  torch.matmul(dprod, v)
        
        output = output.view(batch, self.n_head, n_token, self.token_codim, value_res_x, value_res_y).permute(0,2,1,3,4,5)
        
        if self.nomalizer is not None:
            output = self.non_lin(self.normalize_layer(output))

        
        
        output =  self.end_block(output.view(batch, -1,value_res_x, value_res_y), output_shape = output_shape)
        
        #print(output.shape)
        
        return output

