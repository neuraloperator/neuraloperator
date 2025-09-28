import torch
import torch.nn as nn

from neuralop.models import FNO
from neuralop.layers.channel_mlp import ChannelMLP as NeuralopMLP
from neuralop.layers.spectral_convolution import SpectralConv


class OTNO(FNO):
    def __init__(
            self,
            n_modes,
            hidden_channels,
            in_channels=4,
            out_channels=1,
            lifting_channel_ratio=2,
            projection_channel_ratio=2,
            n_layers=4,
            positional_embedding=None,
            use_mlp=False,
            mlp_expansion=None,
            mlp_dropout=0,
            non_linearity=torch.nn.functional.gelu,
            norm=None,
            preactivation=False,
            fno_skip="linear",
            mlp_skip="soft-gating",
            separable=False,
            factorization=None,
            rank=1,
            joint_factorization=False,
            fixed_rank_modes=False,
            implementation="factorized",
            decomposition_kwargs=dict(),
            domain_padding=None,
            SpectralConv=SpectralConv,
            **kwargs
    ):        
        super().__init__(
            n_modes = n_modes,
            hidden_channels = hidden_channels,
            in_channels = in_channels,
            out_channels = out_channels,
            lifting_channel_ratio = lifting_channel_ratio,
            projection_channel_ratio = projection_channel_ratio,
            n_layers = n_layers,
            positional_embedding=positional_embedding,
            use_channel_mlp = use_mlp,
            channel_mlp_expansion = mlp_expansion,
            channel_mlp_dropout = mlp_dropout,
            channel_mlp_skip = mlp_skip,
            non_linearity = non_linearity,
            norm = norm,
            preactivation = preactivation,
            fno_skip = fno_skip,
            separable = separable,
            factorization = factorization,
            rank = rank,
            joint_factorization = joint_factorization,
            fixed_rank_modes = fixed_rank_modes,
            implementation = implementation,
            decomposition_kwargs = decomposition_kwargs,
            domain_padding = domain_padding,
            SpectralConv = SpectralConv,
            **kwargs
        )

        self.projection = NeuralopMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.hidden_channels*projection_channel_ratio,
            non_linearity=non_linearity,
            n_dim=1,
        )

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    # trans: (1, in_channels, n_s_sqrt, n_s_sqrt)
    def forward(self, trans, ind_dec, **kwargs):
        """TFNO's forward pass"""
        trans = self.lifting(trans)

        if self.domain_padding is not None:
            trans = self.domain_padding.pad(trans)

        for layer_idx in range(self.n_layers):
            trans = self.fno_blocks(trans, layer_idx)

        if self.domain_padding is not None:
            trans = self.domain_padding.unpad(trans) # (1, hidden_channels, n_s_sqrt, n_s_sqrt)

        trans = trans.reshape(self.hidden_channels, -1).permute(1, 0) # (n_s, hidden_channels)

        out = trans[ind_dec].permute(1,0)

        out = out.unsqueeze(0)
        out = self.projection(out).squeeze(1)
        return out