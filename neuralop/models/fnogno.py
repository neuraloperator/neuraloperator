from functools import partial
import torch
import torch.nn.functional as F
import time

from torch import nn

from .fno import FNO

from ..layers.mlp import MLP
from ..layers.embeddings import PositionalEmbedding
from ..layers.spectral_convolution import SpectralConv
from ..layers.integral_transform import IntegralTransform
from ..layers.neighbor_search import NeighborSearch



class FNOGNO(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            nbr_caching=False,
            projection_channels=256,
            gno_coord_dim=3,
            gno_coord_embed_dim=None,
            gno_radius=0.033,
            gno_mlp_hidden_layers=[512, 256],
            gno_mlp_non_linearity=F.gelu, 
            gno_transform_type='linear',
            gno_weighting_fn=None,
            gno_wt_fn_scale=1,
            gno_use_open3d=False,
            fno_n_modes=(16, 16, 16), 
            fno_hidden_channels=64,
            fno_lifting_channels=256,
            fno_projection_channels=256,
            fno_n_layers=4,
            fno_output_scaling_factor=None,
            fno_incremental_n_modes=None,
            fno_block_precision='full',
            fno_use_mlp=False, 
            fno_mlp_dropout=0, 
            fno_mlp_expansion=0.5,
            fno_non_linearity=F.gelu,
            fno_stabilizer=None, 
            fno_norm=None,
            fno_ada_in_features=None,
            fno_ada_in_dim=1,
            fno_preactivation=False,
            fno_skip='linear',
            fno_mlp_skip='soft-gating',
            fno_separable=False,
            fno_factorization=None,
            fno_rank=1.0,
            fno_joint_factorization=False, 
            fno_fixed_rank_modes=False,
            fno_implementation='factorized',
            fno_decomposition_kwargs=dict(),
            fno_domain_padding=None,
            fno_domain_padding_mode='one-sided',
            fno_fft_norm='forward',
            fno_SpectralConv=SpectralConv,
            **kwargs
        ):
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.gno_coord_dim = gno_coord_dim
        if self.gno_coord_dim != 3 and gno_use_open3d:
            print(f'Warning: GNO expects {self.gno_coord_dim}-d data but Open3d expects 3-d data')

        self.in_coord_dim = len(fno_n_modes)
        if self.in_coord_dim != self.gno_coord_dim:
            print(f'Warning: FNO expects {self.in_coord_dim}-d data while GNO expects {self.gno_coord_dim}-d data')

        self.in_coord_dim_forward_order = list(range(self.in_coord_dim))
        self.in_coord_dim_reverse_order = [j + 1 for j in self.in_coord_dim_forward_order]

        if fno_norm == "ada_in":
            if fno_ada_in_features is not None:
                self.adain_pos_embed = PositionalEmbedding(fno_ada_in_features)
                self.ada_in_dim = fno_ada_in_dim*fno_ada_in_features
            else:
                self.ada_in_dim = fno_ada_in_dim
        else:
            self.adain_pos_embed = None
            self.ada_in_dim = None

        self.fno = FNO(
                n_modes=fno_n_modes,
                hidden_channels=fno_hidden_channels,
                in_channels=in_channels + self.in_coord_dim, 
                out_channels=fno_hidden_channels,
                lifting_channels=fno_lifting_channels,
                projection_channels=fno_projection_channels,
                n_layers=fno_n_layers,
                output_scaling_factor=fno_output_scaling_factor,
                incremental_n_modes=fno_incremental_n_modes,
                fno_block_precision=fno_block_precision,
                use_mlp=fno_use_mlp,
                mlp={"expansion": fno_mlp_expansion, "dropout": fno_mlp_dropout},
                non_linearity=fno_non_linearity,
                stabilizer=fno_stabilizer, 
                norm=fno_norm,
                ada_in_features=self.ada_in_dim,
                preactivation=fno_preactivation,
                fno_skip=fno_skip,
                mlp_skip=fno_mlp_skip,
                separable=fno_separable,
                factorization=fno_factorization,
                rank=fno_rank,
                joint_factorization=fno_joint_factorization, 
                fixed_rank_modes=fno_fixed_rank_modes,
                implementation=fno_implementation,
                decomposition_kwargs=fno_decomposition_kwargs,
                domain_padding=fno_domain_padding,
                domain_padding_mode=fno_domain_padding_mode,
                fft_norm=fno_fft_norm,
                SpectralConv=fno_SpectralConv,
                **kwargs
        )
        del self.fno.projection

        self.nb_search_out = NeighborSearch(use_open3d=gno_use_open3d)
        self.gno_radius = gno_radius

        self.gno_weighting_fn = gno_weighting_fn
        sq_radius = self.gno_radius ** 2
        if self.gno_weighting_fn == "linear":
            self.gno_weighting_fn = partial(linear_cutoff, radius=sq_radius, scale=gno_wt_fn_scale)
        elif self.gno_weighting_fn == "bump":
            self.gno_weighting_fn = partial(bump_cutoff, radius=sq_radius, scale=gno_wt_fn_scale)
        elif self.gno_weighting_fn == "tanh":
            self.gno_weighting_fn = partial(tanh_cutoff, radius=sq_radius, scale=gno_wt_fn_scale)
        elif self.gno_weighting_fn == "cubic":
            self.gno_weighting_fn = partial(cubic_cutoff, radius=sq_radius, scale=gno_wt_fn_scale)
        elif self.gno_weighting_fn == "cos":
            self.gno_weighting_fn = partial(cos_cutoff, radius=sq_radius, scale=gno_wt_fn_scale)
        elif self.gno_weighting_fn == "quadr":
            self.gno_weighting_fn = partial(quadr_cutoff, radius=sq_radius, scale=gno_wt_fn_scale)
        elif self.gno_weighting_fn == "bump_sqrt":
            self.gno_weighting_fn = partial(bump_sqrt_cutoff, radius=sq_radius, scale=gno_wt_fn_scale)
        elif self.gno_weighting_fn == "quartic":
            self.gno_weighting_fn = partial(quartic_cutoff, radius=sq_radius, scale=gno_wt_fn_scale)
        elif self.gno_weighting_fn == "quartic_sqrt":
            self.gno_weighting_fn = partial(quartic_sqrt_cutoff, radius=sq_radius, scale=gno_wt_fn_scale)
        elif self.gno_weighting_fn == "octic":
            self.gno_weighting_fn = partial(octic_cutoff, radius=sq_radius, scale=gno_wt_fn_scale)
        elif self.gno_weighting_fn == "octic_sqrt":
            self.gno_weighting_fn = partial(octic_sqrt_cutoff, radius=sq_radius, scale=gno_wt_fn_scale)
        elif self.gno_weighting_fn is not None:
            raise NotImplementedError

        if gno_coord_embed_dim is not None:
            self.pos_embed = PositionalEmbedding(gno_coord_embed_dim)
            self.gno_coord_dim_embed = gno_coord_dim*gno_coord_embed_dim
        else:
            self.pos_embed = None
            self.gno_coord_dim_embed = gno_coord_dim
        

        kernel_in_dim = 2 * self.gno_coord_dim_embed 
        kernel_in_dim += fno_hidden_channels if gno_transform_type != 'linear' else 0

        gno_mlp_hidden_layers.insert(0, kernel_in_dim)
        gno_mlp_hidden_layers.append(fno_hidden_channels)

        self.gno = IntegralTransform(
                    mlp_layers=gno_mlp_hidden_layers,
                    mlp_non_linearity=gno_mlp_non_linearity,
                    transform_type=gno_transform_type,
        )

        self.projection = MLP(in_channels=fno_hidden_channels, 
                              out_channels=out_channels, 
                              hidden_channels=projection_channels, 
                              n_layers=2, 
                              n_dim=1, 
                              non_linearity=fno_non_linearity) 

        # save cached neighbors
        self.nbr_caching = nbr_caching
        self.cached_nbrs = None
        self.domain_lengths = None

    # out_p : (n_out, gno_coord_dim)
    # in_p : (n_1, n_2, ..., n_k, k)
    # f : (n_1, n_2, ..., n_k,  in_channels)
    # ada_in : (fno_ada_in_dim, )

    #returns: (fno_hidden_channels, n_1, n_2, ...)
    def latent_embedding(self, in_p, f, ada_in=None):

        if f is not None:
            in_p = torch.cat((f, in_p), dim=-1)
            in_p = in_p.permute(self.in_coord_dim, *self.in_coord_dim_forward_order).unsqueeze(0)

        # todo: make this general to handle any dim and batch_size
        if self.gno_coord_dim == 2:
            if len(in_p.shape) == 3:
                in_p = in_p.permute(2, 0, 1).unsqueeze(0)
            else:
                in_p = in_p.permute(2, 3, 0, 1)

        #Update Ada IN embedding
        if ada_in is not None:
            if self.adain_pos_embed is not None:
                ada_in_embed = self.adain_pos_embed(ada_in)
            else:
                ada_in_embed = ada_in

            self.fno.fno_blocks.set_ada_in_embeddings(ada_in_embed)

        in_p = self.fno(in_p)
        #Apply FNO blocks
        in_p = self.fno.lifting(in_p)
        if self.fno.domain_padding is not None:
            in_p = self.fno.domain_padding.pad(in_p)

        for layer_idx in range(self.fno.n_layers):
            in_p = self.fno.fno_blocks(in_p, layer_idx)

        if self.fno.domain_padding is not None:
            in_p = self.fno.domain_padding.unpad(in_p)
        
        return in_p.squeeze(0)

    def integrate_latent_batch(self, in_p_batched, out_p_batched, latent_embed_batched):
        # todo: this is extremely inefficient. There are several ways to make it more efficient.

        output_batched = torch.zeros((in_p_batched.shape[2], out_p_batched.shape[1], 1), device=in_p_batched.device)
        compute_norm = self.gno_weighting_fn is not None

        for i in range(in_p_batched.shape[2]):
            # todo: hard-coded for 2-d case
            in_p = in_p_batched[:, :, i, :]
            out_p = out_p_batched[i]
            latent_embed = latent_embed_batched[i, :, :, :]

            if self.nbr_caching:
                if not self.cached_nbrs:
                    print('computing neighbors for the only time')
                    self.cached_nbrs = self.nb_search_out(
                        in_p.view(-1, in_p.shape[-1]), 
                        out_p, 
                        self.gno_radius, 
                        compute_norm=None,
                    )
                in_to_out_nb = self.cached_nbrs
                if compute_norm:
                    in_to_out_nb = self.nb_search_out.compute_norm_separate(
                        self.cached_nbrs, 
                        in_p.view(-1, in_p.shape[-1]),
                        out_p
                        )
            else:
                in_to_out_nb = self.nb_search_out(
                    in_p.view(-1, in_p.shape[-1]), 
                    out_p,
                    self.gno_radius,
                    compute_norm=compute_norm,
                    )

            #Embed input points
            n_in = in_p.view(-1, in_p.shape[-1]).shape[0]
            if self.pos_embed is not None:
                in_p_embed = self.pos_embed(in_p.reshape(-1, )).reshape((n_in, -1))
            else:
                in_p_embed = in_p.reshape((n_in, -1))
            
            #Embed output points
            n_out = out_p.shape[0]
            if self.pos_embed is not None:
                out_p_embed = self.pos_embed(out_p.reshape(-1, )).reshape((n_out, -1))
            else:
                out_p_embed = out_p #.reshape((n_out, -1))

            latent_embed = latent_embed.permute(*self.in_coord_dim_reverse_order, 0).reshape(-1, self.fno.hidden_channels)

            #(n_out, fno_hidden_channels)
            out = self.gno(y=in_p_embed, 
                        neighbors=in_to_out_nb,
                        x=out_p_embed,
                        f_y=latent_embed,
                        weighting_fn=self.gno_weighting_fn)

            out = out.unsqueeze(0).permute(0, 2, 1)

            # Project pointwise to out channels
            #(n_in, out_channels)
            out = self.projection(out).squeeze(0).permute(1, 0)  
            output_batched[i, :, :] = out
        
        return output_batched

    def integrate_latent(self, in_p, out_p, latent_embed):
        #Compute integration region for each output point

        if self.gno_coord_dim == 2 and len(in_p.shape) == 4: # if 2D and batch>1:
            # todo: hard-coded, and we shouldn't separate based on batch_size=1 vs. batch_size>1
            return self.integrate_latent_batch(in_p, out_p, latent_embed)

        # squeeze out_p since batch size is 1
        out_p = out_p.squeeze(0)

        # for each point in out_p, compute neighbors gno_radius apart from in_p
        compute_norm = self.gno_weighting_fn is not None
        if self.nbr_caching:
            if not self.cached_nbrs:
                print('computing neighbors for the only time')
                self.cached_nbrs = self.nb_search_out(
                    in_p.view(-1, in_p.shape[-1]), 
                    out_p, 
                    self.gno_radius, 
                    compute_norm=False,
                )
            in_to_out_nb = self.cached_nbrs
            if compute_norm:
                in_to_out_nb = self.nb_search_out.compute_norm_separate(
                    self.cached_nbrs, 
                    in_p.view(-1, in_p.shape[-1]),
                    out_p
                    )
        else:
            in_to_out_nb = self.nb_search_out(
                in_p.view(-1, in_p.shape[-1]), 
                out_p, 
                self.gno_radius, 
                compute_norm=compute_norm,
            )

        #Embed input points
        n_in = in_p.view(-1, in_p.shape[-1]).shape[0]
        if self.pos_embed is not None:
            in_p_embed = self.pos_embed(in_p.reshape(-1, )).reshape((n_in, -1))
        else:
            in_p_embed = in_p.reshape((n_in, -1))
        
        #Embed output points
        n_out = out_p.shape[0]
        if self.pos_embed is not None:
            out_p_embed = self.pos_embed(out_p.reshape(-1, )).reshape((n_out, -1))
        else:
            out_p_embed = out_p #.reshape((n_out, -1))
        
        #(n_1*n_2*..., fno_hidden_channels)
        #latent_embed = latent_embed.reshape(self.fno.hidden_channels, -1).t()
        #latent_embed = latent_embed.permute(1, 2, 3, 0).reshape(-1, self.fno.hidden_channels)
        latent_embed = latent_embed.permute(*self.in_coord_dim_reverse_order, 0).reshape(-1, self.fno.hidden_channels)

        #(n_out, fno_hidden_channels)
        out = self.gno(
            y=in_p_embed, 
            neighbors=in_to_out_nb,
            x=out_p_embed,
            f_y=latent_embed,
            weighting_fn=self.gno_weighting_fn
        )

        out = out.unsqueeze(0).permute(0, 2, 1)

        # Project pointwise to out channels
        #(n_in, out_channels)
        out = self.projection(out).squeeze(0).permute(1, 0)  
        
        return out


    def forward(self, in_p, out_p, f=None, ada_in=None):
        """forward call of the FNOGNO model

        Parameters
        ----------
        in_p : torch.Tensor
            input geometry, a lattice.
            At every point on the lattice, we expect an optional stack of data channels
            on top of a positional encoding of dimension that matches length of discr_sizes. 
            expects shape batch x (discretization_size 1 x ....)_n x (in channels + pos encoding dim)
        out_p : torch.Tensor
            output points
            expects shape batch x n output points x output dim
        f : _type_, optional
            _description_, by default None
        ada_in : _type_, optional
            _description_, by default None

        Returns
        -------
        out : torch.Tensor
            predicted values (n output points x output channels)
        """
        input_shape = in_p.shape
        assert len(input_shape) == self.gno_coord_dim + 2, "Error: expected data with dimensions matching \
            a batch, input data lattice dimensions, and optional data channels plus positional encoding"
        assert input_shape[-1] == self.in_channels + self.gno_coord_dim, "Error: last dim of \
            input data is expected to be stacked input and positional encoding channels"
        

        # permute b x dim 1 x ... x dim n x in_channels + n --> dim 1 x ... x dim n x b x in_channels + n
        permute_shape = list(range(2,self.gno_coord_dim+2)) + [0,-1]
        in_p = in_p.permute(*permute_shape)

        #Compute latent space embedding
        latent_embed = self.latent_embedding(in_p=in_p, 
                                             f=f, 
                                             ada_in=ada_in)
        
        # todo: make this general to handle any dim and batch size

        # just grab positional encoding of in_p, which is indexed along the last dimension 
        positional_encoding_inds = tuple([slice(None) for _ in range(len(input_shape) - 1)] + [slice(self.in_channels,None,None)])
        in_p = in_p[positional_encoding_inds]

        #Integrate latent space
        out = self.integrate_latent(in_p=in_p, 
                                    out_p=out_p, 
                                    latent_embed=latent_embed)
  
        return out

# TODO(jberner): normalize by volume?
def bump_cutoff(x, radius=1., scale=1., eps=1e-7):
    out = x.clip(0., radius) / radius
    out = - 1 / ((1 - out ** 2) + eps)
    return out.exp() * torch.e * scale

def bump_sqrt_cutoff(x, radius=1., scale=1., eps=1e-7):
    out = - 1 / (1 - x / radius + eps)
    return out.exp() * torch.e * scale

def linear_cutoff(x, radius=1., scale=1.):
    x = (radius - x).clip(0., radius)
    return x * scale / radius

# TODO(jberner): Tanh gives NaNs for the first derivative at 0. and `radius`
def tanh_cutoff(x, radius=1., scale=1., slope=2, eps=1e-6):
    out = x.clip(0., radius) / radius
    out = slope * (2 * out - 1) / (2 * torch.sqrt((1 - out) * out) + eps)
    out = - 0.5 * torch.nn.functional.tanh(out) + 0.5
    return out * scale

def cos_cutoff(x, radius=1., scale=1.):
    x = x / radius
    return scale * (0.5 * torch.cos(torch.pi * x) + 0.5)

def quadr_cutoff(x, radius=1., scale=1.):
    x = x / radius
    left = 1 - 2 * x ** 2
    right = 2 * (1 - x) ** 2
    return scale * torch.where(x < 0.5, left, right)

def cubic_cutoff(x, radius=1., scale=1.):
    b = 3 * scale / (radius ** 2)
    a = 2 * b / (3 * radius)
    out = a * x ** 3 - b * x ** 2 + scale
    assert (x < radius + 0.001).all()
    assert (x > -0.001).all()
    assert (out > -0.001).all()
    return out

def quartic_cutoff(x, radius=1., scale=1.):
    a = scale / radius ** 4
    c = - 2 * scale / radius ** 2 
    return a * x ** 4 + c * x ** 2 + scale

def quartic_sqrt_cutoff(x, radius=1., scale=1.):
    a = scale / radius ** 2
    c = - 2 * scale / radius
    return a * x ** 2 + c * x + scale

def octic_cutoff(x, radius=1., scale=1.):
    x = x / radius
    return scale * (-3 * x ** 8 + 8 * x ** 6 - 6 * x ** 4  + 1)

def octic_sqrt_cutoff(x, radius=1., scale=1.):
    x = x / radius
    return scale * (-3 * x ** 4 + 8 * x ** 3 - 6 * x ** 2  + 1)