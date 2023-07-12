from torch import nn
import torch.nn.functional as F
import torch
from .net_utils import PositionalEmbedding, AdaIN, MLP
from .neigbor_ops import NeighborSearchLayer, NeighborMLPConvLayerLinear
from .integral_transform import IntegralTransform
from neuralop.models import FNO
from .tfno import Projection


class FNOGNO(nn.Module):
    def __init__(
            self,
            in_channels=5,
            out_channels=1,
            fno_modes=(32, 32, 32),
            fno_hidden_channels=86,
            fno_domain_padding=0.125,
            fno_norm="ada_in",
            fno_factorization="tucker",
            fno_rank=0.4,
            adain_embed_dim=64,
            coord_embed_dim=16,
            radius=0.033,
            linear_kernel=True,
            device="cuda:0",
    ):
        
        super().__init__()
        
        if fno_norm == "ada_in":
            init_norm = 'group_norm'
        else:
            init_norm = fno_norm

        self.linear_kernel = linear_kernel

        self.fno = FNO(
            fno_modes,
            hidden_channels=fno_hidden_channels,
            in_channels=in_channels,
            out_channels=1,
            use_mlp=True,
            mlp={"expansion": 1.0, "dropout": 0},
            domain_padding=fno_domain_padding,
            factorization=fno_factorization,
            norm=init_norm,
            rank=fno_rank,
        ).to(device)

        if fno_norm == "ada_in":
            self.adain_pos_embed = PositionalEmbedding(adain_embed_dim).to(device)
            self.fno.fno_blocks.norm = nn.ModuleList(
                AdaIN(adain_embed_dim, fno_hidden_channels)
                for _ in range(self.fno.fno_blocks.n_norms * self.fno.fno_blocks.convs.n_layers)
            ).to(device)
            self.use_adain = True
        else:
            self.use_adain = False

        self.nb_search_out = NeighborSearchLayer(radius=radius).to(device)
        self.pos_embed = PositionalEmbedding(coord_embed_dim).to(device)

        kernel_in_dim = 6 * coord_embed_dim
        kernel_in_dim += 0 if self.linear_kernel else fno_hidden_channels

        self.mlp = MLP([kernel_in_dim, 512, 256, fno_hidden_channels], nn.GELU)
        self.gno = IntegralTransform(mlp=self.mlp).to(device)

        self.projection = Projection(
            in_channels=fno_hidden_channels,
            out_channels=out_channels,
            hidden_channels=256,
            non_linearity=nn.functional.gelu,
            n_dim=1,
        ).to(device)
        
    
    # x_in : (n_in, 3)
    # x_out : (n_x, n_y, n_z, 3)
    # df : (1, n_x, n_y, n_z)
    def forward(self, x_in, x_out, df):
        out_to_in_nb = self.nb_search_out(x_out.view(-1, 3), x_in)
        n_out = x_out.view(-1, 3).shape[0]
        x_out_embed = self.pos_embed(x_out.reshape(-1, )).reshape((n_out, -1))
        # Latent space and distance
        
        x_out = torch.cat((df, x_out.permute(3, 0, 1, 2)), dim=0).unsqueeze(0)  # (1, 12, n_x, n_y, n_z)

        x_out = self.fno.lifting(x_out)
        if self.fno.domain_padding is not None:
            x_out = self.fno.domain_padding.pad(x_out)

        for layer_idx in range(self.fno.n_layers):
            x_out = self.fno.fno_blocks(x_out, layer_idx)

        if self.fno.domain_padding is not None:
            x_out = self.fno.domain_padding.unpad(x_out)

        x_out = x_out.squeeze(0).permute(1, 2, 3, 0).reshape(-1, self.fno.hidden_channels)
        # x_out: (n_x*n_y*n_z, fno_hidden_channels)
        n_in = x_in.shape[0]
        x_in_embed = self.pos_embed(x_in.reshape(-1, )).reshape((n_in, -1))
        x_out = self.gno(x_out_embed, out_to_in_nb, x_in_embed, x_out)
        
        x_out = x_out.unsqueeze(0).permute(0, 2, 1)
        # Project pointwise to out channels
        x_out = self.projection(x_out).squeeze(0).permute(1, 0)  # (n_in, out_channels)
        return x_out
    