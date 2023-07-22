from torch import nn
import torch.nn.functional as F
import torch
from .embeddings import PositionalEmbedding
from .spectral_convolution import FactorizedSpectralConv
from .integral_transform import IntegralTransform
from .neighbor_search import NeighborSearch
from .tfno import FNO
from .tfno import Projection


class FNOGNO(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            projection_channels=256,
            gno_coord_embed_dim=None,
            gno_radius=0.033,
            gno_mlp_hidden_layers=[512, 256],
            gno_mlp_non_linearity=F.gelu, 
            gno_transform_type=0,
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
            fno_SpectralConv=FactorizedSpectralConv,
            **kwargs
        ):
        
        super().__init__()
        #Must be 3 due to Open3D
        #Pad 1D and 2D data with zeros
        gno_coord_dim = 3

        self.fno = FNO(
                n_modes=fno_n_modes,
                hidden_channels=fno_hidden_channels,
                in_channels=in_channels + 3, 
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
                ada_in_features=fno_ada_in_features,
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

        if fno_norm == "ada_in":
            self.adain_pos_embed = PositionalEmbedding(fno_ada_in_features)
        else:
            self.adain_pos_embed = None

        self.nb_search_out = NeighborSearch()
        self.gno_radius = gno_radius

        if gno_coord_embed_dim is not None:
            self.pos_embed = PositionalEmbedding(gno_coord_embed_dim)
            self.gno_coord_dim = gno_coord_dim*gno_coord_embed_dim
        else:
            self.pos_embed = None
            self.gno_coord_dim = gno_coord_dim
        

        kernel_in_dim = 2 * self.gno_coord_dim 
        kernel_in_dim += fno_hidden_channels if gno_transform_type != 0 else 0

        gno_mlp_hidden_layers.insert(0, kernel_in_dim)
        gno_mlp_hidden_layers.append(fno_hidden_channels)

        self.gno = IntegralTransform(
                    mlp_layers=gno_mlp_hidden_layers,
                    mlp_non_linearity=gno_mlp_non_linearity,
                    transform_type=gno_transform_type 
        )

        self.projection = Projection(
            in_channels=fno_hidden_channels,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            non_linearity=fno_non_linearity,
            n_dim=1,
        )

    # out_p : (n_out, 3)
    # in_p : (n_1, n_2, ..., 3)
    # f : (n_1, n_2, ...,  in_channels)
    # ada_in : (1, )

    #returns: (fno_hidden_channels, n_1, n_2, ...)
    def latent_embedding(self, in_p, f, ada_in=None):
        #Input shape
        n_comb_channels = in_p.shape[-1] + f.shape[-1] 
        in_p_shape = tuple(in_p.shape[0:-1])
        
        #(1, gno_coord_dim + in_channels, n_1, n_2, ...)
        in_p = torch.cat((f, in_p), dim=-1).view(-1, n_comb_channels).t()
        in_p = in_p.view(n_comb_channels, *in_p_shape).unsqueeze(0) 

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
        in_to_out_nb = self.nb_search_out(in_p.view(-1, in_p.shape[-1]), self.gno_radius, out_p)

        #Embed input points
        n_in = in_p.view(-1, in_p.shape[-1]).shape[0]
        if self.pos_embed is not None:
            in_p_embed = self.pos_embed(in_p.reshape(-1, )).reshape((n_in, -1))
        else:
            in_p_embed = in_p.reshape((n_in, -1))
        
        #Embed output points
        n_out = out_p.view(-1, out_p.shape[-1]).shape[0]
        if self.pos_embed is not None:
            out_p_embed = self.pos_embed(out_p.reshape(-1, )).reshape((n_out, -1))
        else:
            out_p_embed = out_p.reshape((n_out, -1))
        
        #(n_1*n_2*..., fno_hidden_channels)
        latent_embed = latent_embed.reshape(self.fno.hidden_channels, -1).t()

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


    def forward(self, in_p, out_p, f, ada_in=None):
        
        #Compute latent space embedding
        latent_embed = self.latent_embedding(in_p=in_p, 
                                             f=f, 
                                             ada_in=ada_in)
        #Integrate latent space
        out = self.integrate_latent(in_p=in_p, 
                                    out_p=out_p, 
                                    latent_embed=latent_embed)
        
        return out