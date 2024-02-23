from functools import partialmethod

import torch.nn as nn
import torch.nn.functional as F

from ..layers.attention_kernel_integral import AttentionKernelIntegral
from ..layers.mlp import MLPLinear
from ..layers.embeddings import SirenNet, GaussianFourierFeatureTransform, RotaryEmbedding


class TransformerNO(nn.Module):
    """N-Dimensional Transformer-based Neural Operator (currently does not support N>2)
        using softmax-free attention to compute the kernel integral.

        Each layer in the encoder part is organized as follow (a.k.a pre-norm version of Transformer layer):
            u = attn(norm(u)) + u
            u = mlp(norm(u)) + u
        where u is the input function to the layer.

        For the decoder (cross-attention), given query bases q and src function u:
            u_out = attn(q, u)   # u_out will has the same shape as q

        Parameters
        ----------
        n_dim : int
            Number of dimensions of the domain
        in_channels : int, optional
            Number of input channels, by default 1
        out_channels : int, optional
            Number of output channels, by default 1
        encoder_hidden_channels : int
            Width of the encoder (i.e. number of channels in attention layer and MLP)
        decoder_hidden_channels : int
            Width of the decoder (i.e. number of channels in attention layer and MLP)
        encoder_num_heads: int, optional
            Number of heads in the encoder attention, by default 1
        decoder_num_heads: int, optional
            Number of heads in the decoder cross-attention, by default 8
        encoder_head_n_channels: int, optional
            Dimension of each attention head in the encoder, by default equals to encoder_hidden_channels
        decoder_head_n_channels: int, optional
            Dimension of each attention head in the decoder, by default equals to decoder_hidden_channels
        encoder_n_layers : int, optional
            Number of Transformer layer in the encoder, by default 3
        query_basis: string, optional
            Type of coordinate-based network to compute query basis function in the decoder,
            by default 'siren', other options are ['fourier', 'linear']
        query_siren_layers: int, optional
            Number of layers in SirenNet, by default 4
        query_fourier_scale: float, optional
            Scale of the Gaussian Fourier Feature Transform in random Fourier Feature, by default 2.0
        use_mlp : bool, optional
            Whether to use an MLP layer after each attention block, by default True
        mlp_dropout : float , optional
            droupout parameter of MLP layer, by default 0
        mlp_expansion : float, optional
            expansion parameter of MLP layer, by default 2.0
        non_linearity : nn.Module, optional
            Non-Linearity module to use, by default F.gelu
        norm: string, optional
            Normalization module to use, by default layernorm

        """
    def __init__(self,
                 n_dim,
                 in_channels=1,
                 out_channels=1,
                 encoder_hidden_channels=128,
                 decoder_hidden_channels=128,
                 encoder_num_heads=1,
                 decoder_num_heads=8,
                 encoder_head_n_channels=None,
                 decoder_head_n_channels=None,
                 encoder_n_layers=3,
                 query_basis='siren',
                 query_siren_layers=4,          # number of layers in SirenNet
                 query_fourier_scale=2.0,       # scale of the Gaussian Fourier Feature Transform
                 pos_emb='rotary',              # ['rotary', 'none']
                 use_mlp=True,
                 mlp_dropout=0,
                 mlp_expansion=2.0,
                 non_linearity=F.gelu,
                 norm='layer_norm',      # ['layer_norm', 'instance_norm', ''group_norm', 'none']
                ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dim = n_dim
        self.encoder_num_heads = encoder_num_heads
        self.decoder_num_heads = decoder_num_heads
        self.encoder_hidden_channels = encoder_hidden_channels
        self.decoder_hidden_channels = decoder_hidden_channels
        self.encoder_head_n_channels = encoder_head_n_channels if encoder_head_n_channels is not None else encoder_hidden_channels
        self.decoder_head_n_channels = decoder_head_n_channels if decoder_head_n_channels is not None else decoder_hidden_channels
        self.encoder_n_layers = encoder_n_layers
        self.query_basis = query_basis
        self.query_siren_layers = query_siren_layers
        self.query_fourier_scale = query_fourier_scale
        self.pos_emb = pos_emb

        self.use_mlp = use_mlp
        self.mlp_dropout = mlp_dropout
        self.mlp_expansion = mlp_expansion
        self.non_linearity = non_linearity
        self.norm = norm

        if self.query_basis not in ['siren', 'fourier', 'linear']:
            raise ValueError(f'query_basis must be one of ["siren", "fourier", "linear"], got {self.query_basis}')

        if self.pos_emb not in ['rotary', 'none']:
            raise ValueError(f'pos_emb must be one of ["rotary", "none"], got {self.pos_emb}')

        if self.norm not in ['layer_norm', 'instance_norm', 'group_norm', 'none']:
            raise ValueError(f'norm must be one of ["layer_norm", "instance_norm", "group_norm", "none"], got {self.norm}')

        if self.pos_emb == 'rotary':
            self.enc_pos_emb_module = RotaryEmbedding(self.encoder_head_n_channels // self.n_dim)
            self.dec_pos_emb_module = RotaryEmbedding(self.decoder_head_n_channels // self.n_dim)
        else:
            self.enc_pos_emb_module = None
            self.dec_pos_emb_module = None

        # top and bottom layer
        self.lifting = nn.Linear(self.in_channels, self.encoder_hidden_channels)
        self.projection = MLPLinear([self.decoder_hidden_channels, self.decoder_hidden_channels, self.out_channels],
                                    non_linearity=self.non_linearity)

        self.enc_to_dec = nn.Linear(self.encoder_hidden_channels, self.decoder_hidden_channels) \
            if self.encoder_hidden_channels != self.decoder_hidden_channels else nn.Identity()

        # build encoder
        self.encoder_blocks = nn.ModuleList([])
        for layer in range(self.encoder_n_layers):
            encoder_layer = nn.ModuleList([])
            encoder_layer.append(self.get_normalization(self.norm, self.encoder_hidden_channels))
            encoder_layer.append(AttentionKernelIntegral(self.encoder_hidden_channels,
                                                         self.encoder_hidden_channels,
                                                         self.encoder_num_heads,
                                                         self.encoder_head_n_channels))
            if self.use_mlp:
                encoder_layer.append(self.get_normalization(self.norm, self.encoder_hidden_channels))
                encoder_layer.append(MLPLinear([self.encoder_hidden_channels,
                                                int(self.encoder_hidden_channels*mlp_expansion),
                                                self.encoder_hidden_channels],
                                               non_linearity=self.non_linearity,
                                               dropout=self.mlp_dropout))
            self.encoder_blocks.append(encoder_layer)

        # build basis for decoder
        if self.query_basis == 'siren':
            self.query_basis_fn = SirenNet(dim_in=self.n_dim,
                                           dim_hidden=self.decoder_hidden_channels,
                                           dim_out=self.decoder_num_heads*self.decoder_head_n_channels,
                                           num_layers=self.query_siren_layers)
        elif self.query_basis == 'fourier':
            self.query_basis_fn = GaussianFourierFeatureTransform(self.n_dim,
                                                                  mapping_size=self.decoder_num_heads*self.decoder_head_n_channels//2,
                                                                  scale=self.query_fourier_scale)
        elif self.query_basis == 'linear':
            self.query_basis_fn = nn.Linear(self.n_dim, self.decoder_num_heads*self.decoder_head_n_channels)

        self.decoder = AttentionKernelIntegral(self.decoder_hidden_channels,
                                               self.decoder_hidden_channels,
                                               self.decoder_num_heads,
                                               self.decoder_head_n_channels,
                                               project_query=False)   # query projection is done by query_basis_fn

        self.projection_norm = self.get_normalization(self.norm, self.decoder_hidden_channels)

    @staticmethod
    def get_normalization(norm, channels):
        if norm == 'none':
            norm_fn = nn.Identity()
        elif norm == "instance_norm":
            norm_fn = nn.InstanceNorm1d(channels)
        elif norm == "group_norm":
            norm_fn = nn.GroupNorm(num_groups=32, num_channels=channels)
        elif norm == 'layer_norm':
            norm_fn = nn.LayerNorm(channels)
        else:
            raise ValueError(
                f"Got norm={norm} but expected None or one of "
                "[instance_norm, group_norm, layer_norm]"
            )
        return norm_fn

    def normalize(self, u, norm_fn):
        if isinstance(norm_fn, nn.GroupNorm) or isinstance(norm_fn, nn.InstanceNorm1d):
            u = u.permute(0, 2, 1).contiguous()   # channel first
            u = norm_fn(u)
            u = u.permute(0, 2, 1).contiguous()
        else:
            u = norm_fn(u)
        return u

    def transformer_layer_forward(self,
                                  u,
                                  pos,
                                  pos_emb_module,
                                  transformer_block):
        if self.use_mlp:
            [norm_attn, attn, norm_ffn, ffn] = transformer_block
            u = attn(self.normalize(u, norm_attn), pos, positional_embedding_module=pos_emb_module) + u
            u = ffn(self.normalize(u, norm_ffn)) + u
        else:
            [norm_attn, attn] = transformer_block
            u = attn(self.normalize(u, norm_attn), pos, positional_embedding_module=pos_emb_module) + u
        return u

    def forward(self,
                u,
                pos_src,
                pos_qry=None,
                **kwargs):
        """Transformer NO's forward pass,
           please note that coordinates must be normalized to [-1, 1] interval when using siren"""

        # encoder part, use self-attention to process input function
        u = self.lifting(u)
        for transformer_layer in self.encoder_blocks:
            u = self.transformer_layer_forward(u, pos_src, self.enc_pos_emb_module, transformer_layer)
        u = self.enc_to_dec(u)

        # decoder part
        if pos_qry is None:
            pos_qry = pos_src   # assume that the query points are the same as the source points
        query_emb = self.query_basis_fn(pos_qry)
        query_emb = query_emb.view(pos_qry.shape[0], -1, self.decoder_num_heads*self.decoder_head_n_channels)
        u_out = self.decoder(
            u_src=u,
            pos_src=pos_src,
            u_qry=query_emb,
            pos_qry=pos_qry,
            positional_embedding_module=self.dec_pos_emb_module,
            **kwargs
        )
        u_out = self.projection(self.normalize(u_out, self.projection_norm))
        return u_out
