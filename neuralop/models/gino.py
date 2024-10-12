from functools import partial
import torch
import torch.nn.functional as F
import time

from torch import nn


from ..layers.channel_mlp import ChannelMLP
from ..layers.embeddings import SinusoidalEmbedding
from ..layers.fno_block import FNOBlocks
from ..layers.spectral_convolution import SpectralConv
from ..layers.integral_transform import IntegralTransform
from ..layers.neighbor_search import NeighborSearch

class GINO(nn.Module):
    """GINO: Geometry-informed Neural Operator

        Parameters
        ----------
        in_channels : int
            feature dimension of input points
        out_channels : int
            feature dimension of output points
        projection_channels : int, optional
            number of channels in FNO pointwise projection
        gno_coord_dim : int, optional
            geometric dimension of input/output queries, by default 3
        gno_coord_embed_dim : int, optional
            dimension of positional embedding for gno coordinates, by default None
        gno_embed_max_positions : int, optional
            max positions for use in gno positional embedding, by default 10000
        gno_radius : float, optional
            radius in input/output space for GNO neighbor search, by default 0.033
        in_gno_channel_mlp_hidden_layers : list, optional
            widths of hidden layers in input GNO, by default [80, 80, 80]
        out_gno_channel_mlp_hidden_layers : list, optional
            widths of hidden layers in output GNO, by default [512, 256]
        gno_channel_mlp_non_linearity : nn.Module, optional
            nonlinearity to use in gno ChannelMLP, by default F.gelu
        in_gno_transform_type : str, optional
            transform type parameter for input GNO, by default 'linear'
            see neuralop.layers.IntegralTransform
        out_gno_transform_type : str, optional
            transform type parameter for output GNO, by default 'linear'
            see neuralop.layers.IntegralTransform
        gno_use_open3d : bool, optional
            whether to use open3d neighbor search, by default False
            if False, uses pure-PyTorch fallback neighbor search
        gno_use_torch_scatter : bool, optional
            whether to use torch_scatter's neighborhood reduction function
            or the native PyTorch implementation in IntegralTransform layers.
            If False, uses the fallback PyTorch version.
        out_gno_tanh : bool, optional
            whether to use tanh to stabilize outputs of the output GNO, by default False
        fno_in_channels : int, optional
            number of input channels for FNO, by default 26
        fno_n_modes : tuple, optional
            number of modes along each dimension 
            to use in FNO, by default (16, 16, 16)
        fno_hidden_channels : int, optional
            hidden channels for use in FNO, by default 64
        lifting_channels : int, optional
            number of channels in FNO's pointwise lifting, by default 256
        fno_projection_channels : int, optional
            number of channels in FNO's pointwise projection, by default 256
        fno_n_layers : int, optional
            number of layers in FNO, by default 4
        fno_resolution_scaling_factor : float | None, optional
            factor by which to scale output of FNO, by default None
        fno_incremental_n_modes : list[int] | None, defaults to None
        if passed, sets n_modes separately for each FNO layer.
        fno_block_precision : str, defaults to 'full'
            data precision to compute within fno block
        fno_use_channel_mlp : bool, defaults to False
            Whether to use a ChannelMLP layer after each FNO block.
        fno_channel_mlp_dropout : float, defaults to 0
            dropout parameter of above ChannelMLP.
        fno_channel_mlp_expansion : float, defaults to 0.5
            expansion parameter of above ChannelMLP.
        fno_non_linearity : nn.Module, defaults to F.gelu
            nonlinear activation function between each FNO layer.
        fno_stabilizer : nn.Module | None, defaults to None
            By default None, otherwise tanh is used before FFT in the FNO block.
        fno_norm : nn.Module | None, defaults to None
            normalization layer to use in FNO.
        fno_ada_in_features : int | None, defaults to None
            if an adaptive mesh is used, number of channels of its positional embedding.
        fno_ada_in_dim : int, defaults to 1
            dimensions of above FNO adaptive mesh.
        fno_preactivation : bool, defaults to False
            whether to use Resnet-style preactivation.
        fno_skip : str, defaults to 'linear'
            type of skip connection to use.
        fno_channel_mlp_skip : str, defaults to 'soft-gating'
            type of skip connection to use in the FNO
            'linear': conv layer
            'soft-gating': weights the channels of the input
            'identity': nn.Identity
        fno_separable : bool, defaults to False
            if True, use a depthwise separable spectral convolution.
        fno_factorization : str {'tucker', 'tt', 'cp'} |  None, defaults to None
            Tensor factorization of the parameters weight to use
        fno_rank : float, defaults to 1.0
            Rank of the tensor factorization of the Fourier weights.
        fno_joint_factorization : bool, defaults to False
            Whether all the Fourier layers should be parameterized by a single tensor (vs one per layer).
        fno_fixed_rank_modes : bool, defaults to False
            Modes to not factorize.
        fno_implementation : str {'factorized', 'reconstructed'} | None, defaults to 'factorized'
            If factorization is not None, forward mode to use::
            * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass
            * `factorized` : the input is directly contracted with the factors of the decomposition
        fno_decomposition_kwargs : dict, defaults to dict()
            Optionaly additional parameters to pass to the tensor decomposition.
        fno_conv_module : nn.Module, defaults to SpectralConv
            Spectral Convolution module to use.
        """
    def __init__(
            self,
            in_channels,
            out_channels,
            projection_channels=256,
            gno_coord_dim=3,
            gno_coord_embed_dim=None,
            gno_embed_max_positions=100000,
            gno_radius=0.033,
            in_gno_channel_mlp_hidden_layers=[80, 80, 80],
            out_gno_channel_mlp_hidden_layers=[512, 256],
            gno_channel_mlp_non_linearity=F.gelu, 
            in_gno_transform_type='linear',
            out_gno_transform_type='linear',
            gno_use_open3d=False,
            gno_use_torch_scatter=True,
            out_gno_tanh=None,
            fno_in_channels=3,
            fno_n_modes=(16, 16, 16), 
            fno_hidden_channels=64,
            lifting_channels=256,
            fno_n_layers=4,
            fno_resolution_scaling_factor=None,
            fno_incremental_n_modes=None,
            fno_block_precision='full',
            fno_use_channel_mlp=False, 
            fno_channel_mlp_dropout=0,
            fno_channel_mlp_expansion=0.5,
            fno_non_linearity=F.gelu,
            fno_stabilizer=None, 
            fno_norm=None,
            fno_ada_in_features=None,
            fno_ada_in_dim=1,
            fno_preactivation=False,
            fno_skip='linear',
            fno_channel_mlp_skip='soft-gating',
            fno_separable=False,
            fno_factorization=None,
            fno_rank=1.0,
            fno_joint_factorization=False, 
            fno_fixed_rank_modes=False,
            fno_implementation='factorized',
            fno_decomposition_kwargs=dict(),
            fno_conv_module=SpectralConv,
            **kwargs
        ):
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gno_coord_dim = gno_coord_dim
        self.fno_hidden_channels = fno_hidden_channels

        self.lifting_channels = lifting_channels

        # TODO: make sure this makes sense in all contexts
        fno_in_channels = self.in_channels
        self.fno_in_channels = fno_in_channels

        if self.gno_coord_dim != 3 and gno_use_open3d:
            print(f'Warning: GNO expects {self.gno_coord_dim}-d data but Open3d expects 3-d data')

        self.in_coord_dim = len(fno_n_modes)
        self.gno_out_coord_dim = len(fno_n_modes) # gno output and fno will use same dimensions
        if self.in_coord_dim != self.gno_coord_dim:
            print(f'Warning: FNO expects {self.in_coord_dim}-d data while input GNO expects {self.gno_coord_dim}-d data')

        self.in_coord_dim_forward_order = list(range(self.in_coord_dim))
        # channels starting at 2 to permute everything after channel and batch dims
        self.in_coord_dim_reverse_order = [j + 2 for j in self.in_coord_dim_forward_order]

        self.fno_norm = fno_norm
        if self.fno_norm == "ada_in":
            if fno_ada_in_features is not None:
                self.adain_pos_embed = SinusoidalEmbedding(in_channels=fno_ada_in_dim,
                                                           num_frequencies=fno_ada_in_features,
                                                           embedding_type='transformer',
                                                           max_positions=gno_embed_max_positions)
                self.ada_in_dim = self.adain_pos_embed.out_channels
            else:
                self.ada_in_dim = fno_ada_in_dim
                self.adain_pos_embed = None
        else:
            self.adain_pos_embed = None
            self.ada_in_dim = None

        self.lifting = ChannelMLP(in_channels=self.fno_in_channels,
                                  hidden_channels=lifting_channels,
                                  out_channels=fno_hidden_channels,
                                  n_layers=3)
        
        self.fno_blocks = FNOBlocks(
                n_modes=fno_n_modes,
                hidden_channels=fno_hidden_channels,
                in_channels=fno_hidden_channels,
                out_channels=fno_hidden_channels,
                positional_embedding=None,
                n_layers=fno_n_layers,
                resolution_scaling_factor=fno_resolution_scaling_factor,
                incremental_n_modes=fno_incremental_n_modes,
                fno_block_precision=fno_block_precision,
                use_channel_mlp=fno_use_channel_mlp,
                channel_mlp_expansion=fno_channel_mlp_expansion,
                channel_mlp_dropout=fno_channel_mlp_dropout,
                non_linearity=fno_non_linearity,
                stabilizer=fno_stabilizer, 
                norm=fno_norm,
                ada_in_features=self.ada_in_dim,
                preactivation=fno_preactivation,
                fno_skip=fno_skip,
                channel_mlp_skip=fno_channel_mlp_skip,
                separable=fno_separable,
                factorization=fno_factorization,
                rank=fno_rank,
                joint_factorization=fno_joint_factorization, 
                fixed_rank_modes=fno_fixed_rank_modes,
                implementation=fno_implementation,
                decomposition_kwargs=fno_decomposition_kwargs,
                domain_padding=None,
                domain_padding_mode=None,
                conv_module=fno_conv_module,
                **kwargs
        )

        self.nb_search_out = NeighborSearch(use_open3d=gno_use_open3d)
        self.gno_radius = gno_radius
        self.out_gno_tanh = out_gno_tanh

        if gno_coord_embed_dim is not None:
            self.pos_embed = SinusoidalEmbedding(in_channels=gno_coord_dim,
                                                 num_frequencies=gno_coord_embed_dim, 
                                                 embedding_type='transformer',
                                                 max_positions=gno_embed_max_positions)
            self.gno_coord_dim_embed = self.pos_embed.out_channels
        else:
            self.pos_embed = None
            self.gno_coord_dim_embed = self.gno_out_coord_dim
        

        ### input GNO
        # input to the first GNO ChannelMLP: x pos encoding, y (integrand) pos encoding
        in_kernel_in_dim = self.gno_coord_dim * 2
        # add f_y features if input GNO uses a nonlinear kernel
        if in_gno_transform_type == "nonlinear" or in_gno_transform_type == "nonlinear_kernelonly":
            in_kernel_in_dim += self.in_channels
            
        in_gno_channel_mlp_hidden_layers.insert(0, in_kernel_in_dim)
        in_gno_channel_mlp_hidden_layers.append(fno_in_channels) 
        self.gno_in = IntegralTransform(
            channel_mlp_layers=in_gno_channel_mlp_hidden_layers,
            channel_mlp_non_linearity=gno_channel_mlp_non_linearity,
            transform_type=in_gno_transform_type,
            use_torch_scatter=gno_use_torch_scatter
        )

        ### output GNO
        out_kernel_in_dim = 2 * self.gno_coord_dim_embed 
        out_kernel_in_dim += fno_hidden_channels if out_gno_transform_type != 'linear' else 0
        out_gno_channel_mlp_hidden_layers.insert(0, out_kernel_in_dim)
        out_gno_channel_mlp_hidden_layers.append(fno_hidden_channels)
        self.gno_out = IntegralTransform(
                    channel_mlp_layers=out_gno_channel_mlp_hidden_layers,
                    channel_mlp_non_linearity=gno_channel_mlp_non_linearity,
                    transform_type=out_gno_transform_type,
                    use_torch_scatter=gno_use_torch_scatter
        )

        self.projection = ChannelMLP(in_channels=fno_hidden_channels, 
                              out_channels=self.out_channels, 
                              hidden_channels=projection_channels, 
                              n_layers=2, 
                              n_dim=1, 
                              non_linearity=fno_non_linearity) 


    # out_p : (n_out, gno_coord_dim)

    #returns: (fno_hidden_channels, n_1, n_2, ...)
    def latent_embedding(self, in_p, ada_in=None):

        # in_p : (batch, n_1 , ... , n_k, in_channels + k)
        # ada_in : (fno_ada_in_dim, )

        # permute (b, n_1, ..., n_k, c) -> (b,c, n_1,...n_k)
        in_p = in_p.permute(0, len(in_p.shape)-1, *list(range(1,len(in_p.shape)-1)))
        #Update Ada IN embedding    
        if ada_in is not None:
            if ada_in.ndim == 2:
                ada_in = ada_in.squeeze(0)
            if self.adain_pos_embed is not None:
                ada_in_embed = self.adain_pos_embed(ada_in.unsqueeze(0)).squeeze(0)
            else:
                ada_in_embed = ada_in
            if self.fno_norm == "ada_in":
                self.fno_blocks.set_ada_in_embeddings(ada_in_embed)

        #Apply FNO blocks
        in_p = self.lifting(in_p)

        for idx in range(self.fno_blocks.n_layers):
            in_p = self.fno_blocks(in_p, idx)

        return in_p 

    def integrate_latent(self, in_p, out_p, latent_embed):
        batch_size = latent_embed.shape[0]

        # output shape: (batch, n_out, out_channels) or (n_out, out_channels)
        in_to_out_nb = self.nb_search_out(
            in_p.view(-1, in_p.shape[-1]), 
            out_p,
            self.gno_radius,
            )
    
        #Embed input points
        n_in = in_p.view(-1, in_p.shape[-1]).shape[0]
        if self.pos_embed is not None:
            in_p_embed = self.pos_embed(in_p.reshape((n_in, -1)))
        else:
            in_p_embed = in_p.reshape((n_in, -1))
        
        #Embed output points
        n_out = out_p.shape[0]
        if self.pos_embed is not None:
            out_p_embed = self.pos_embed(out_p.reshape((n_out, -1)))
        else:
            out_p_embed = out_p #.reshape((n_out, -1))
                
        #latent_embed shape b, c, n_1, n_2, ..., n_k
        latent_embed = latent_embed.permute(0, *self.in_coord_dim_reverse_order, 1).reshape(batch_size, -1, self.fno_hidden_channels)
        # shape b, n_out, channels
        if self.out_gno_tanh in ['latent_embed', 'both']:
            latent_embed = torch.tanh(latent_embed)
        
        #(n_out, fno_hidden_channels)
        out = self.gno_out(y=in_p_embed, 
                    neighbors=in_to_out_nb,
                    x=out_p_embed,
                    f_y=latent_embed,)
                    #weighting_fn=self.gno_weighting_fn)
        out = out.permute(0, 2, 1)
        # Project pointwise to out channels
        #(b, n_in, out_channels)
        out = self.projection(out).permute(0, 2, 1)  

        return out
    
    def forward(self,  x, input_geom, latent_queries, output_queries, ada_in=None, **kwargs):
        """forward pass of GNO --> latent embedding w/FNO --> GNO out

        Parameters
        ----------
        x : torch.Tensor
            input function a defined on the input domain `input_geom`
            shape (batch, n_in, in_channels) 
        input_geom : torch.Tensor
            input domain coordinate mesh
            shape (1, n_in, gno_coord_dim)
        latent_queries : torch.Tensor
            latent geometry on which to compute FNO latent embeddings
            a grid on [0,1] x [0,1] x ....
            shape (1, n_gridpts_1, .... n_gridpts_n, gno_coord_dim)
        output_queries : torch.Tensor
            points at which to query the final GNO layer to get output
            shape (batch, n_out, gno_coord_dim)
        ada_in : torch.Tensor, optional
            adaptive scalar instance parameter, defaults to None
        """
        batch_size = x.shape[0]

        input_geom = input_geom.squeeze(0) 
        latent_queries = latent_queries.squeeze(0)
    
        spatial_nbrs = self.nb_search_out(input_geom, 
                                          latent_queries.view((-1, latent_queries.shape[-1])), 
                                          radius=self.gno_radius)
        
        in_p = self.gno_in(y=input_geom,
                           x=latent_queries.view((-1, latent_queries.shape[-1])),
                           f_y=x,
                           neighbors=spatial_nbrs)
        
        grid_shape = latent_queries.shape[:-1] # disregard positional encoding dim
        
        # shape (batch, grid1, ...gridn, fno_in_channels)
        in_p = in_p.view((batch_size, *grid_shape, self.fno_in_channels))
        
        # take apply fno in latent space
        latent_embed = self.latent_embedding(in_p=in_p, 
                                             ada_in=ada_in)

        # Integrate latent space to output queries
        out = self.integrate_latent(latent_queries, output_queries, latent_embed)

        return out