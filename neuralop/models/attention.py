from .fno_block import FNOBlocks
import torch.nn as nn
<<<<<<< HEAD
from .spectral_convolution import FactorizedSpectralConv, SpectralConvKernel2d
=======
from .spectral_convolution import FactorizedSpectralConv
>>>>>>> 1ea96afd429a01e9b67da7ad9fdefcb187ef6a1d
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from .fino import SpectralConvKernel2d
import torch

class TnoBlock2d(nn.Module):
<<<<<<< HEAD
    def __init__(self,in_channels, n_modes,
                 n_head = 1,
                 token_codim = 1,
=======
    def __init__(self, n_modes,
                 n_head=1,
                 token_codim=1,
>>>>>>> 1ea96afd429a01e9b67da7ad9fdefcb187ef6a1d
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
<<<<<<< HEAD
                 fft_norm='forward', normalizer = None, **kwarg):
=======
                 fft_norm='forward', **kwarg):
>>>>>>> 1ea96afd429a01e9b67da7ad9fdefcb187ef6a1d
        
        super().__init__()
        self.token_codim = token_codim
        self.n_head = n_head
        self.output_scaling_factor = output_scaling_factor
        
        mixer_modes = [i//self.n_head for i in n_modes]
<<<<<<< HEAD
        self.K = FNOBlocks(in_channels= self.token_codim, out_channels= self.n_head * self.token_codim, n_modes= n_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor = 1/n_head,non_linearity=non_linearity,\
=======
        self.K = FNOBlocks(in_channels=self.token_codim, out_channels= self.n_head * self.token_codim, n_modes= mixer_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor=1/n_head,non_linearity= lambda x: x,\
>>>>>>> 1ea96afd429a01e9b67da7ad9fdefcb187ef6a1d
                                            norm=norm, preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip,mlp_dropout=0, mlp_expansion=0.5,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                            SpectralConv= SpectralConv,n_layers=1)

<<<<<<< HEAD
        self.Q = FNOBlocks(in_channels= self.token_codim, out_channels= self.n_head * self.token_codim, n_modes= n_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor = 1/n_head,non_linearity=non_linearity,\
=======
        self.Q = FNOBlocks(in_channels= self.token_codim, out_channels= self.n_head * self.token_codim, n_modes= mixer_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor=1/n_head,non_linearity=lambda x: x,\
>>>>>>> 1ea96afd429a01e9b67da7ad9fdefcb187ef6a1d
                                            norm=norm, preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip, mlp_dropout=0, mlp_expansion=0.5,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                            SpectralConv= SpectralConv, n_layers=1)

<<<<<<< HEAD
        self.V = FNOBlocks(in_channels= self.token_codim, out_channels= self.n_head * self.token_codim, n_modes= n_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor = 1/n_head,non_linearity=non_linearity,\
=======
        self.V = FNOBlocks(in_channels= self.token_codim, out_channels= self.n_head * self.token_codim, n_modes= mixer_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor=1/n_head,non_linearity= lambda x: x,\
                                            norm=norm, preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip, mlp_dropout=0, mlp_expansion=0.5,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                            SpectralConv= SpectralConv,n_layers=1)
        
        #if n_head != 1:
        self.mixer = FNOBlocks(in_channels= self.n_head * self.token_codim, out_channels= self.token_codim, n_modes= mixer_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor=n_head,non_linearity=lambda x: x,\
>>>>>>> 1ea96afd429a01e9b67da7ad9fdefcb187ef6a1d
                                            norm=norm, preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip, mlp_dropout=0, mlp_expansion=0.5,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
<<<<<<< HEAD
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
=======
                                            SpectralConv= SpectralConv,n_layers=1) 
        
#         self.end_block = FNOBlocks(in_channels= in_channels, out_channels= in_channels, n_modes= n_modes,\
#                                             use_mlp=use_mlp, mlp=mlp, output_scaling_factor=self.output_scaling_factor,non_linearity=non_linearity,\
#                                             norm='instance_norm', preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip, mlp_dropout=0, mlp_expansion=0.5,\
#                                             incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
#                                             fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
#                                             factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
#                                             SpectralConv= SpectralConv,n_layers=1)

        
    def forward(self, x, output_shape=None):
        batch, n_token , in_res_x, in_res_y=x.shape[0], x.shape[1]//self.token_codim, x.shape[-2], x.shape[-1]
>>>>>>> 1ea96afd429a01e9b67da7ad9fdefcb187ef6a1d
        
        
        assert x.shape[1]%self.token_codim == 0
        
<<<<<<< HEAD
        x = x.view(-1, self.token_codim, in_res_x, in_res_y)
        #print(x.shape)
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)
        
        #print("vales and keys 0", v.shape, k.shape)
=======
        xa = rearrange(x, 'b (t d) h w -> b t d h w', d=self.token_codim)
        xa = rearrange(xa, 'b t d h w -> (b t) d h w')
        #print(x.shape)
        k = self.K(xa)
        q = self.Q(xa)
        v = self.V(xa)
>>>>>>> 1ea96afd429a01e9b67da7ad9fdefcb187ef6a1d
        
        res_x, res_y = k.shape[-2], k.shape[-1]
        value_res_x, value_res_y = v.shape[-2], v.shape[-1]
        
<<<<<<< HEAD
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
        
        
=======
        k = rearrange(k, '(b t) (a d) h w -> b a t (d h w)', b=batch, a=self.n_head )
        q = rearrange(q, '(b t) (a d) h w -> b a t (d h w)', b=batch, a=self.n_head )
        v = rearrange(v, '(b t) (a d) h w -> b a t (d h w)', b=batch, a=self.n_head )
>>>>>>> 1ea96afd429a01e9b67da7ad9fdefcb187ef6a1d


<<<<<<< HEAD
        dprod =  torch.matmul(q, k)/k.shape[-1]

        dprod = F.softmax(dprod, dim = -1)
                
        
        v = v.view(batch, self.n_head, n_token, -1)

        output =  torch.matmul(dprod, v)
        
        output = output.view(batch,self.n_head,  n_token, self.token_codim, value_res_x, value_res_y).permute(0,2,1,3,4,5)
        
        output = output.reshape(-1, self.n_head,  self.token_codim,  value_res_x, value_res_y).view(-1, self.n_head*self.token_codim, value_res_x, value_res_y)
        
        if self.n_head != 1:
            output = self.mixer(output, output_shape = (in_res_x, in_res_y))
        

        
        output = output.view(batch, n_token, self.token_codim, in_res_x, in_res_y).view(batch, -1, in_res_x, in_res_y)
        
        
        if self.nomalizer is not None:
            output = self.non_lin(self.normalize_layer(output))


        
        output =  self.end_block(output, output_shape = output_shape)
=======
        dprod = torch.matmul(q, k.transpose(-1, -2))/k.shape[-1]

        dprod = F.softmax(dprod, dim=-1)
                
        output =  torch.matmul(dprod, v)
        
        output = rearrange(output, 'b a t (d h w) -> b t a d h w', d=self.token_codim, h=value_res_x, w=value_res_y)
        
#         if self.n_head != 1:
        output = rearrange(output, 'b t a d h w -> (b t) (a d) h w')
        output = self.mixer(output, output_shape=(in_res_x, in_res_y))
        output = rearrange(output, '(b t) d h w -> b (t d) h w', b=batch)
#         else:
#             output = output = rearrange(output, 'b t a d h w -> b (t a d) h w')
        
        output = output+x
        
#         output =  self.end_block(output, output_shape=output_shape)
>>>>>>> 1ea96afd429a01e9b67da7ad9fdefcb187ef6a1d
        
        return output