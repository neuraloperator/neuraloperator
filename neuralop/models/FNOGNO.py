from torch import nn
import torch.nn.functional as F
import torch
from .spectral_convolution import FactorizedSpectralConv
from .skip_connections import skip_connection
from .resample import resample
from .mlp import MLP, PositionalEmbedding, AdaIN
# from .normalization_layers import AdaIN
from .fno_block import FNOBlocks
from .integral_transform import IntegralTransform, NeighborSearch
from .tfno import FNO, Projection

# A simple feedforward neural network
class MLP(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(MLP, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x
    

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
            max_in_points=5000,
    ):
        
        super().__init__()

        if fno_norm == "ada_in":
            init_norm = 'group_norm'
        else:
            init_norm = fno_norm

        self.linear_kernel = linear_kernel
        self.max_in_points = max_in_points

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
        )

        if fno_norm == "ada_in":
            self.adain_pos_embed = PositionalEmbedding(adain_embed_dim)
            self.fno.fno_blocks.norm = nn.ModuleList(
                AdaIN(adain_embed_dim, fno_hidden_channels)
                for _ in range(self.fno.fno_blocks.n_norms * self.fno.fno_blocks.convs.n_layers)
            )
            self.use_adain = True
        else:
            self.use_adain = False

        self.nb_search_out = NeighborSearch()
        self.pos_embed = PositionalEmbedding(coord_embed_dim)

        kernel_in_dim = 6 * coord_embed_dim
        kernel_in_dim += 0 if self.linear_kernel else fno_hidden_channels

        self.mlp = MLP([kernel_in_dim, 512, 256, fno_hidden_channels], nn.GELU)
        
        self.gno = IntegralTransform(self.mlp)

        self.projection = Projection(
            in_channels=fno_hidden_channels,
            out_channels=out_channels,
            hidden_channels=256,
            non_linearity=nn.functional.gelu,
            n_dim=1,
        )

        self.fno_hidden_channels = fno_hidden_channels
        self.device = nn.Parameter(torch.empty(0)).device
    
    # x_in : (n_in, 3)
    # x_out : (n_x, n_y, n_z, 3)
    # df : (1, n_x, n_y, n_z)
    def forward(self, x_in, x_out, df, radius=0.055):
        import pdb; pdb.set_trace()
        out_to_in_nb = self.nb_search_out(x_out.view(-1, 3), radius, x_in)

        n_out = x_out.view(-1, 3).shape[0]
        x_out_embed = self.pos_embed(
            x_out.reshape(-1, )
        ).reshape(
            (n_out, -1)
        )
        # Latent space and distance
        x_out = torch.cat(
            (df, x_out.permute(3, 0, 1, 2)), dim=0
        ).unsqueeze(
            0
        )  # (1, 12, n_x, n_y, n_z)

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
        x_in_embed = self.pos_embed(
            x_in.reshape(-1, )
        ).reshape(
            (n_in, -1)
        )
        # y = x_out_embed, neighbors=out_to_in_nb, x=x_in_embed, f_y=x_out
        x_out = self.gno(x_out_embed, out_to_in_nb, x=x_in_embed, f_y=x_out, weights=None)
        x_out = x_out.unsqueeze(0).permute(0, 2, 1)
        # Project pointwise to out channels
        x_out = self.projection(x_out).squeeze(0).permute(1, 0)  # (n_in, out_channels)
        
        return x_out
    
    def data_dict_to_input(self, data_dict):
        x_in = data_dict["centroids"][0]  # (n_in, 3)
        x_out = (
            data_dict["query_points"].squeeze(0).permute(1, 2, 3, 0)
        )  # (n_x, n_y, n_z, 3)
        df = data_dict["distance"]  # (1, n_x, n_y, n_z)

        # info_fields = torch.cat([
        #    v*torch.ones_like(df) for _, v in data_dict['info'][0].items()
        # ], dim=0)

        info_fields = data_dict['inlet_velocity'] * torch.ones_like(df)

        df = torch.cat((
            df, info_fields
        ), dim=0)

        if self.use_adain:
            vel = torch.tensor([data_dict['inlet_velocity']]).view(-1, ).to(self.device)
            vel_embed = self.adain_pos_embed(vel)
            for norm in self.fno.fno_blocks.norm:
                norm.update_embeddding(vel_embed)

        x_in, x_out, df = (
            x_in.to(self.device),
            x_out.to(self.device),
            df.to(self.device),
        )
        return x_in, x_out, df
    
    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        x_in, x_out, df = self.data_dict_to_input(data_dict)

        if self.max_in_points is not None:
            r = min(self.max_in_points, x_in.shape[0])
            pred_chunks = []
            x_in_chunks = torch.split(x_in, r, dim=0)
            for j in range(len(x_in_chunks)):
                pred_chunks.append(self(x_in_chunks[j], x_out, df))
            pred = torch.cat(tuple(pred_chunks), dim=0)
        else:
            pred = self(x_in, x_out, df)

        pred = pred.reshape(1, -1)

        if loss_fn is None:
            loss_fn = self.loss
        truth = data_dict["pressure"][0].to(self.device).reshape(1, -1)
        out_dict = {"l2": loss_fn(pred, truth)}

        if decode_fn is not None:
            pred = decode_fn(pred)
            truth = decode_fn(truth)
            out_dict["l2_decoded"] = loss_fn(pred, truth)

            torch.save(pred.view(-1, ).cpu().detach(), 'pred_ahmed_' + str(kwargs['ind']).zfill(3) + '.pt')
        return out_dict

    def loss_dict(self, data_dict, loss_fn=None):
        x_in, x_out, df = self.data_dict_to_input(data_dict)

        if self.max_in_points is not None:
            r = min(self.max_in_points, x_in.shape[0])
            indices = torch.randperm(x_in.shape[0])[:r]
            x_in = x_in[indices, ...]

        pred = self(x_in, x_out, df)

        if loss_fn is None:
            loss_fn = self.loss

        if self.max_in_points is not None:
            truth = data_dict["pressure"][0][indices].view(1, -1).to(self.device)
        else:
            truth = data_dict["pressure"][0].view(1, -1).to(self.device)

        # truth = data_dict["pressure"][0][indices].view(1, -1).to(self.device)
        return {
            "loss": loss_fn(
                pred.view(1, -1), truth
            )
        }


    
    @property
    def incremental_n_modes(self):
        return self._incremental_n_modes

    @incremental_n_modes.setter
    def incremental_n_modes(self, incremental_n_modes):
        self.convs.incremental_n_modes = incremental_n_modes

    def get_block(self, indices):
        """Returns a sub-FNO Block layer from the jointly parametrized main block

        The parametrization of an FNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError('A single layer is parametrized, directly use the main class.')
        
        return SubModule(self, indices)
    
    def __getitem__(self, indices):
        return self.get_block(indices)


class SubModule(nn.Module):
    """Class representing one of the sub_module from the mother joint module

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules, they all point to the same data, 
    which is shared.
    """
    def __init__(self, main_module, indices):
        super().__init__()
        self.main_module = main_module
        self.indices = indices
    
    def forward(self, x):
        return self.main_module.forward(x, self.indices)

