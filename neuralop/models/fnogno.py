import torch
import torch.nn.functional as F

from torch import nn

from .fno import FNO

from ..layers.mlp import MLP
from ..layers.embeddings import PositionalEmbedding
from ..layers.spectral_convolution import SpectralConv
from ..layers.integral_transform import IntegralTransform
from ..layers.neighbor_search import NeighborSearch


class FNOGNO(nn.Module):
    """FNOGNO: Fourier/Geometry Neural Operator 

        Parameters
        ----------
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        projection_channels : int, defaults to 256
             number of hidden channels in embedding block of FNO. 
        gno_coord_dim : int, defaults to 3
            dimension of GNO input data. 
        gno_coord_embed_dim : int | None, defaults to none
            dimension of embeddings of GNO coordinates. 
        gno_radius : float, defaults to 0.033
            radius parameter to construct graph. 
        gno_mlp_hidden_layers : list, defaults to [512, 256]
            dimension of hidden MLP layers of GNO. 
        gno_mlp_non_linearity : nn.Module, defaults to F.gelu
            nonlinear activation function between layers
        gno_transform_type : str, defaults to 'linear'
            type of kernel integral transform to apply in GNO. 
            kernel k(x,y): parameterized as MLP integrated over a neighborhood of x
            options: 'linear_kernelonly': integrand is k(x, y) 
                        'linear' : integrand is k(x, y) * f(y)
                        'nonlinear_kernelonly' : integrand is k(x, y, f(y))
                        'nonlinear' : integrand is k(x, y, f(y)) * f(y)
        gno_use_open3d : bool, defaults to False
            whether to use Open3D functionality
            if False, uses simple fallback neighbor search
        fno_n_modes : tuple, defaults to (16, 16, 16)
            number of modes to keep along each spectral dimension of FNO block
        fno_hidden_channels : int, defaults to 64
            number of hidden channels of fno block. 
        fno_lifting_channels : int, defaults to 256
            dimension of hidden layers in FNO lifting block.
        fno_n_layers : int, defaults to 4
            number of FNO layers in the block. 
        fno_output_scaling_factor : float | None, defaults to None
            factor by which to rescale output predictions in the original domain
        fno_incremental_n_modes : list[int] | None, defaults to None
            if passed, sets n_modes separately for each FNO layer. 
        fno_block_precision : str, defaults to 'full'
            data precision to compute within fno block
        fno_use_mlp : bool, defaults to False
            Whether to use an MLP layer after each FNO block. 
        fno_mlp_dropout : float, defaults to 0
            dropout parameter of above MLP. 
        fno_mlp_expansion : float, defaults to 0.5
            expansion parameter of above MLP. 
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
        fno_mlp_skip : str, defaults to 'soft-gating'
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
        fno_domain_padding : float | None, defaults to None
            If not None, percentage of padding to use. 
        fno_domain_padding_mode : str {'symmetric', 'one-sided'}, defaults to 'one-sided'
            How to perform domain padding. 
        fno_fft_norm : str, defaults to 'forward'
            normalization parameter of torch.fft to use in FNO. Defaults to 'forward'
        fno_SpectralConv : nn.Module, defaults to SpectralConv
             Spectral Convolution module to use.
        """
    
    
    
    def __init__(
            self,
            in_channels,
            out_channels,
            projection_channels=256,
            gno_coord_dim=3,
            gno_coord_embed_dim=None,
            gno_radius=0.033,
            gno_mlp_hidden_layers=[512, 256],
            gno_mlp_non_linearity=F.gelu, 
            gno_transform_type='linear',
            gno_use_open3d=False,
            fno_n_modes=(16, 16, 16), 
            fno_hidden_channels=64,
            fno_lifting_channels=256,
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
                projection_channels=1,
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
                    transform_type=gno_transform_type 
        )

        self.projection = MLP(in_channels=fno_hidden_channels, 
                              out_channels=out_channels, 
                              hidden_channels=projection_channels, 
                              n_layers=2, 
                              n_dim=1, 
                              non_linearity=fno_non_linearity) 

    # out_p : (n_out, gno_coord_dim)
    # in_p : (n_1, n_2, ..., n_k, k)
    # f : (n_1, n_2, ..., n_k,  in_channels)
    # ada_in : (fno_ada_in_dim, )

    #returns: (fno_hidden_channels, n_1, n_2, ...)
    def latent_embedding(self, in_p, f, ada_in=None):
        in_p = torch.cat((f, in_p), dim=-1)
        in_p = in_p.permute(self.in_coord_dim, *self.in_coord_dim_forward_order).unsqueeze(0)

        #Update Ada IN embedding
        if ada_in is not None:
            if self.adain_pos_embed is not None:
                ada_in_embed = self.adain_pos_embed(ada_in)
            else:
                ada_in_embed = ada_in

            self.fno.fno_blocks.set_ada_in_embeddings(ada_in_embed)

        #Apply FNO blocks
        in_p = self.fno.lifting(in_p)
        if self.fno.domain_padding is not None:
            in_p = self.fno.domain_padding.pad(in_p)

        for layer_idx in range(self.fno.n_layers):
            in_p = self.fno.fno_blocks(in_p, layer_idx)

        if self.fno.domain_padding is not None:
            in_p = self.fno.domain_padding.unpad(in_p)
        
        return in_p.squeeze(0)
    
    def integrate_latent(self, in_p, out_p, latent_embed):
        #Compute integration region for each output point
        in_to_out_nb = self.nb_search_out(in_p.view(-1, in_p.shape[-1]), out_p, self.gno_radius)

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
        latent_embed = latent_embed.permute(*self.in_coord_dim_reverse_order, 0).reshape(-1, self.fno.hidden_channels)

        #(n_out, fno_hidden_channels)
        out = self.gno(y=in_p_embed, 
                       neighbors=in_to_out_nb,
                       x=out_p_embed,
                       f_y=latent_embed)
        
        out = out.unsqueeze(0).permute(0, 2, 1)

        # Project pointwise to out channels
        #(n_in, out_channels)
        out = self.projection(out).squeeze(0).permute(1, 0)  
        
        return out


    def forward(self, in_p, out_p, f, ada_in=None, **kwargs):
        
        #Compute latent space embedding
        latent_embed = self.latent_embedding(in_p=in_p, 
                                             f=f, 
                                             ada_in=ada_in)
        #Integrate latent space
        out = self.integrate_latent(in_p=in_p, 
                                    out_p=out_p, 
                                    latent_embed=latent_embed)
        
        return out