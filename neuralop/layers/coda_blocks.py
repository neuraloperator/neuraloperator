from functools import partial
import logging
import numpy as np
import torch
import torch
from torch import nn
import torch.nn.functional as F
from .fno_block import FNOBlocks
from .spectral_convolution import SpectralConv


# Codomain Attention Blocks
class CODABlocks(nn.Module):
    """ Class for the Co-domain Attention Blocks (CODABlocks)

    Args:
        n_modes (list): Number of modes for each dimension  used in K,Q,V operator.
        n_heads (int): Number of heads for the attention mechanism.
        token_codimension (int): Co-dimension of each variable or number of channels associated with each variables.
        output_scaling_factor (float): Scaling factor for the output.
        incremental_n_modes (list): Incremental number of modes for each dimension (for incremental training).
        head_codimension (int): Co-dimension of each of output token for each head.

        # config for FNO_blocks used as K,Q,V operator
        use_channel_mlp (bool): whether to use mlp layers to parameterize skip connections, by default False
        non_linearity (callable): Non-linearity function to be used.
        preactivation (bool): whether to use preactivation, by default False.
        fno_skip (str): Type of skip connection to be used, by default 'linear'.
        mlp_skip (str):  module to use for ChannelMLP skip connections, by default "soft-gating".
        channel_mlp_expansion (float): expansion parameter for self.channel_mlp, by default 0.5.
        separable (bool): whether to use separable convolutions, by default False.
        factorization (str): Type of factorization to be used, by default 'tucker'.
        rank (float): Rank of the factorization, by default 1.0.
        spectral_convolution (callable): Spectral convolution module to be used.
        joint_factorization (bool): whether to factorize all spectralConv weights as one tensor, by default False

        # Normalization
        Normalizer (callable): Normalization module to be used.

        codimension_size (int): Size of the codimension for whole function, only used for permutation_eq = False.
        per_channel_attention (bool): whether to use per channel attention, by default True (overwrite token_codimension to 1).
        permutation_eq (bool): whether to use permutation equivariant mixer layer  after attention mechanism.
        temperature (float): Temperature parameter for attention mechanism.
        kqv_non_linear (bool): whether to use non-linear activation for K,Q,V operator.
    """

    def __init__(
        self,
        n_modes,
        n_heads=1,
        token_codimension=1,
        output_scaling_factor=None,
        incremental_n_modes=None,
        head_codimension=None,
        use_channel_mlp=False,
        non_linearity=F.gelu,
        preactivation=False,
        fno_skip='linear',
        channel_mlp_skip='soft-gating',
        channel_mlp_expansion=1.0,
        separable=False,
        factorization='tucker',
        rank=1.0,
        spectral_convolution=None,
        Normalizer=None,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation='factorized',
        decomposition_kwargs=None,
        fft_norm='forward',
        codimension_size=None,
        per_channel_attention=True,
        permutation_eq=True,
        temperature=1.0,
        kqv_non_linear=False,
        **_kwargs,
    ):
        super().__init__()

        # Co-dimension of each variable/token. The token embedding space is
        # identical to the variable space, so their dimensionalities are equal.
        self.token_codimension = token_codimension

        # (maybe) codim of attention from each head
        self.head_codimension = (head_codimension
                                 if head_codimension is not None
                                 else token_codimension)
        self.n_heads = n_heads  # number of heads
        self.output_scaling_factor = output_scaling_factor
        self.temperature = temperature

        # K,Q,V operator with or without non_lin

        if kqv_non_linear:
            kqv_activation = non_linearity
        else:
            kqv_activation = torch.nn.Identity()

        self.permutation_eq = permutation_eq

        self.codimension_size = codimension_size
        self.mixer_token_codimension = token_codimension

        if per_channel_attention:
            # for per channel attention, forcing the values of token dims
            self.token_codimension = 1
            self.head_codimension = 1

        # this scale used for downsampling Q,K functions
        scale = 2 if per_channel_attention else 1
        scale = min(self.n_heads, scale)

        mixer_modes = [i // scale for i in n_modes]

        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        common_args = dict(
            use_channel_mlp=use_channel_mlp,
            preactivation=preactivation,
            channel_mlp_skip=channel_mlp_skip,
            mlp_dropout=0,
            incremental_n_modes=incremental_n_modes,
            rank=rank,
            fft_norm=fft_norm,
            channel_mlp_expansion=channel_mlp_expansion,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
        )

        kqv_args = dict(
            in_channels=self.token_codimension,
            out_channels=self.n_heads * self.head_codimension,
            n_modes=mixer_modes,
            # args below are shared with Projection block
            non_linearity=kqv_activation,
            fno_skip='linear',
            norm=None,
            apply_skip=True,
            n_layers=1,
        )
        self.K = FNOBlocks(
            output_scaling_factor=1 / scale,
            conv_module=spectral_convolution,
            **kqv_args,
            **common_args,
        )
        self.Q = FNOBlocks(
            output_scaling_factor=1 / scale,
            conv_module=spectral_convolution,
            **kqv_args,
            **common_args,
        )
        self.V = FNOBlocks(
            output_scaling_factor=1,
            conv_module=spectral_convolution,
            **kqv_args,
            **common_args,
        )

        if self.n_heads * self.head_codimension != self.token_codimension:
            self.proj = FNOBlocks(
                in_channels=self.n_heads * self.head_codimension,
                out_channels=self.token_codimension,
                n_modes=n_modes,
                output_scaling_factor=1,
                # args below are shared with KQV blocks
                apply_skip=True,
                non_linearity=torch.nn.Identity(),
                fno_skip='linear',
                norm=None,
                conv_module=spectral_convolution,
                n_layers=1,
                **common_args,
            )
        else:
            self.proj = None

        self.attention_normalizer = Normalizer(self.token_codimension)

        mixer_args = dict(
            n_modes=n_modes,
            output_scaling_factor=1,
            non_linearity=non_linearity,
            norm='instance_norm',
            fno_skip=fno_skip,
            conv_module=spectral_convolution,
        )
        # We have an option to make the last operator (MLP in regular
        # Transformer block) permutation equivariant. i.e., applying the
        # operator per variable or applying the operator on the whole channel
        # (like regular FNO).
        if permutation_eq:
            self.mixer = FNOBlocks(
                in_channels=self.mixer_token_codimension,
                out_channels=self.mixer_token_codimension,
                apply_skip=True,
                n_layers=2,
                **mixer_args,
                **common_args,
            )
            self.norm1 = Normalizer(self.token_codimension)
            self.norm2 = Normalizer(self.mixer_token_codimension)
            self.mixer_out_normalizer = Normalizer(
                self.mixer_token_codimension)

        else:
            self.mixer = FNOBlocks(
                in_channels=codimension_size,
                out_channels=codimension_size,
                n_layers=2,
                **mixer_args,
                **common_args,
            )
            self.norm1 = Normalizer(codimension_size)
            self.norm2 = Normalizer(codimension_size)
            self.mixer_out_normalizer = Normalizer(codimension_size)

    def forward(self, *args):
        raise NotImplementedError(
            "Use a proper subclass of TNOBlock (i.e. TnoBlock2d or TNOBlock3D).")


class CODABlocks2D(CODABlocks):
    def __init__(self, *args, **kwargs):
        Normalizer = kwargs.get("Normalizer")
        if Normalizer is None:
            Normalizer = partial(nn.InstanceNorm2d, affine=True)
        kwargs["Normalizer"] = Normalizer

        Convolution = kwargs.get("spectral_convolution")
        if Convolution is None:
            Convolution = SpectralConv
        kwargs["spectral_convolution"] = Convolution

        super().__init__(*args, **kwargs)

    def compute_attention(self, xa, batch_size):
        """Compute the key-query-value variant of the attention matrix.

        Assumes input ``xa`` has been normalized.
        """
        k = self.K(xa)
        q = self.Q(xa)
        v = self.V(xa)

        value_x, value_y = v.shape[-2], v.shape[-1]

        assert k.size(
            1) % self.n_heads == 0, "Number of channels in k, q, and v should be divisible by number of heads"

        # reshaping '(b t) (a d) h w -> b a t (d h w)'
        t = k.size(0) // batch_size  # Compute the time dimension `t`
        d = k.size(1) // self.n_heads  # Compute the dimension `d`

        k = k.view(batch_size, t, self.n_heads, d, k.size(-2),
                   k.size(-1))
        q = q.view(batch_size, t, self.n_heads, d, q.size(-2),
                   q.size(-1))
        v = v.view(batch_size, t, self.n_heads, d, v.size(-2),
                   v.size(-1))
        k = k.permute(0, 2, 1, 3, 4, 5).contiguous()
        q = q.permute(0, 2, 1, 3, 4, 5).contiguous()
        v = v.permute(0, 2, 1, 3, 4, 5).contiguous()

        k = k.view(batch_size, self.n_heads, t, d * k.size(-2)
                   * k.size(-1))
        q = q.view(batch_size, self.n_heads, t, d * q.size(-2)
                   * q.size(-1))
        v = v.view(batch_size, self.n_heads, t, d * v.size(-2)
                   * v.size(-1))

        dprod = (torch.matmul(q, k.transpose(-1, -2)) /
                 (np.sqrt(k.shape[-1]) * self.temperature))
        dprod = F.softmax(dprod, dim=-1)

        attention = torch.matmul(dprod, v)

        # Reshape from (b, a, t, d * h * w) to (b, a, t, d, h, w)
        attention = attention.view(
            attention.size(0),
            attention.size(1),
            attention.size(2),
            self.head_codimension,
            value_x,
            value_y)

        # Permute the dimensions to rearrange into (b * t, a * d, h, w)
        attention = attention.permute(
            0, 2, 1, 3, 4, 5).contiguous()  # (b, t, a, d, h, w)

        # Reshape to flatten the first and second dimensions (b * t) and the
        # third and fourth dimensions (a * d)
        attention = attention.view(attention.size(0) * attention.size(1),
                                   attention.size(2) * attention.size(3),
                                   value_x, value_y)  # (b * t, a * d, h, w)

        return attention

    def forward(self, x):
        if self.permutation_eq:
            return self._forward_equivariant(x)
        else:
            return self._forward_non_equivariant(x)

    def _forward_equivariant(self, x):
        '''
        uses permutation equivariant mixer layer after attention mechanism. We share the same mixer layer
        for all the varibales.
        '''
        batch_size = x.shape[0]
        output_shape = x.shape[-2:]

        assert x.shape[1] % self.token_codimension == 0, "Number of channels in x should be divisible by token_codimension"

        # reshape 'b (t d) h w -> (b t) d h w'
        t = x.size(1) // self.token_codimension
        xa = x.view(
            x.size(0),
            t,
            self.token_codimension,
            x.size(2),
            x.size(3))  # (b, t, d, h, w)

        xa = xa.permute(0, 1, 2, 3, 4).contiguous()
        xa = xa.view(
            x.size(0) * t,
            self.token_codimension,
            x.size(2),
            x.size(3))  # (b * t, d, h, w)
        xa_norm = self.norm1(xa)
        attention = self.compute_attention(xa_norm, batch_size)
        if self.proj is not None:
            attention = self.proj(attention)

        attention = self.attention_normalizer(attention + xa)
        attention_normalized = self.norm2(attention)
        output = self.mixer(attention_normalized, output_shape=output_shape)

        output = self.mixer_out_normalizer(output) + attention

        # reshaped '(b t) d h w -> b (t d) h w'
        t = output.size(0) // batch_size
        output = output.view(
            batch_size,
            t,
            output.size(1),
            output.size(2),
            output.size(3))
        output = output.view(
            batch_size,
            t * output.size(2),
            output.size(3),
            output.size(4))

        return output

    def _forward_non_equivariant(self, x):
        '''
        Uses non-permutation equivariant mixer layer after attention mechanism.
        '''
        batch_size = x.shape[0]
        output_shape = x.shape[-2:]

        assert x.shape[1] % self.token_codimension == 0

        x_norm = self.norm1(x)
        # Compute time dimension t
        t = x_norm.size(1) // self.token_codimension
        xa = x_norm.view(
            x_norm.size(0),
            t,
            self.token_codimension,
            x_norm.size(2),
            x_norm.size(3))  # (b, t, d, h, w)
        xa = xa.view(
            x_norm.size(0) * t,
            self.token_codimension,
            x_norm.size(2),
            x_norm.size(3))  # (b * t, d, h, w)

        attention = self.compute_attention(xa, batch_size)
        if self.proj is not None:
            attention = self.proj(attention)

        # (b t) d h w -> b (t d) h w
        t = attention.size(0) // batch_size  # Compute the time dimension `t`
        attention = attention.view(batch_size, t, attention.size(
            1), attention.size(2), attention.size(3))  # (b, t, d, h, w)
        attention = attention.view(
            batch_size,
            t * attention.size(2),
            attention.size(3),
            attention.size(4))  # (b, t * d, h, w)

        attention_normalized = self.norm2(attention)
        output = self.mixer(attention_normalized, output_shape=output_shape)

        return output


class CODABlocks3D(CODABlocks):
    def __init__(self, *args, **kwargs):
        Normalizer = kwargs.get("Normalizer")
        if Normalizer is None:
            Normalizer = partial(nn.InstanceNorm3d, affine=True)
        kwargs["Normalizer"] = Normalizer

        Convolution = kwargs.get("spectral_convolution")
        if Convolution is None:
            Convolution = SpectralConv
        kwargs["spectral_convolution"] = Convolution

        super().__init__(*args, **kwargs)

    def compute_attention(self, xa, batch_size):
        """Compute the key-query-value variant of the attention matrix.

        Assumes input ``xa`` has been normalized.
        """
        k = self.K(xa)
        q = self.Q(xa)
        v = self.V(xa)

        v_duration, v_height, v_width = v.shape[-3:]

        # (b k) (a d) t h w -> b a k (d t h w)
        k_size = k.size(0) // batch_size
        d_size = k.size(1) // self.n_heads
        k = k.view(
            batch_size,
            k_size,
            self.n_heads,
            d_size,
            k.size(2),
            k.size(3),
            k.size(4))  # (b, k, a, d, t, h, w)
        q = q.view(
            batch_size,
            k_size,
            self.n_heads,
            d_size,
            q.size(2),
            q.size(3),
            q.size(4))  # (b, k, a, d, t, h, w)
        v = v.view(
            batch_size,
            k_size,
            self.n_heads,
            d_size,
            v.size(2),
            v.size(3),
            v.size(4))  # (b, k, a, d, t, h, w)
        k = k.permute(0, 2, 1, 3, 4, 5, 6).contiguous()
        q = q.permute(0, 2, 1, 3, 4, 5, 6).contiguous()
        v = v.permute(0, 2, 1, 3, 4, 5, 6).contiguous()
        k = k.view(
            batch_size,
            self.n_heads,
            k_size,
            d_size *
            k.size(4) *
            k.size(5) *
            k.size(6))  # (b, a, k, d * t * h * w)
        q = q.view(
            batch_size,
            self.n_heads,
            k_size,
            d_size *
            q.size(4) *
            q.size(5) *
            q.size(6))  # (b, a, k, d * t * h * w)
        v = v.view(
            batch_size,
            self.n_heads,
            k_size,
            d_size *
            v.size(4) *
            v.size(5) *
            v.size(6))  # (b, a, k, d * t * h * w)

        dprod = (torch.matmul(q, k.transpose(-1, -2)) /
                 (self.temperature * np.sqrt(k.shape[-1])))
        dprod = F.softmax(dprod, dim=-1)

        attention = torch.matmul(dprod, v)

        # 'b a k (d t h w) -> (b k) (a d) t h w',
        attention = attention.view(
            attention.size(0),  # b
            attention.size(1),  # a
            attention.size(2),  # k
            self.head_codimension,  # d
            v_duration,          # t
            v_height,            # h
            v_width              # w
        )
        attention = attention.permute(
            0, 2, 1, 3, 4, 5, 6).contiguous()  # (b, k, a, d, t, h, w)
        attention = attention.view(
            attention.size(0) * attention.size(1),  # (b * k)
            attention.size(2) * attention.size(3),  # (a * d)
            attention.size(4),                      # t
            attention.size(5),                      # h
            attention.size(6)                       # w
        )

        return attention

    def forward(self, x):
        if self.permutation_eq:
            return self._forward_equivariant(x)
        else:
            return self._forward_non_equivariant(x)

    def _forward_equivariant(self, x):
        batch_size = x.shape[0]
        output_shape = x.shape[-3:]

        assert x.shape[1] % self.token_codimension == 0

        # 'b (k d) t h w -> (b k) d t h w'
        k_size = x.size(1) // self.token_codimension
        xa = x.view(
            x.size(0),
            k_size,
            self.token_codimension,
            x.size(2),
            x.size(3),
            x.size(4))
        xa = xa.permute(0, 1, 2, 3, 4, 5).contiguous()
        xa = xa.view(
            x.size(0) * k_size,
            self.token_codimension,
            x.size(2),
            x.size(3),
            x.size(4))

        xa_norm = self.norm1(xa)

        attention = self.compute_attention(xa_norm, batch_size)
        if self.proj is not None:
            attention = self.proj(attention)

        attention = self.attention_normalizer(attention + xa)
        attention_normalized = self.norm2(attention)
        output = self.mixer(attention_normalized, output_shape=output_shape)

        output = self.mixer_out_normalizer(output) + attention
        # '(b k) d t h w -> b (k d) t h w'
        k_size = output.size(0) // batch_size
        output = output.view(
            batch_size,
            k_size,
            output.size(1),
            output.size(2),
            output.size(3),
            output.size(4))  # (b, k, d, t, h, w)
        output = output.view(
            batch_size,
            k_size * output.size(2),
            output.size(3),
            output.size(4),
            output.size(5))  # (b, k * d, t, h, w)

        return output

    def _forward_non_equivariant(self, x):
        batch_size = x.shape[0]
        output_shape = x.shape[-3:]

        assert x.shape[1] % self.token_codimension == 0

        x_norm = self.norm1(x)
        # 'b (k d) t h w -> (b k) d t h w'
        k_size = x_norm.size(1) // self.token_codimension
        xa = x_norm.view(
            x_norm.size(0),
            k_size,
            self.token_codimension,
            x_norm.size(2),
            x_norm.size(3),
            x_norm.size(4))  # (b, k, d, t, h, w)
        xa = xa.view(
            x_norm.size(0) * k_size,
            self.token_codimension,
            x_norm.size(2),
            x_norm.size(3),
            x_norm.size(4))  # (b * k, d, t, h, w)

        attention = self.compute_attention(xa, batch_size)
        if self.proj is not None:
            attention = self.proj(attention)

        # '(b k) d t h w -> b (k d) t h w'
        k_size = output.size(0) // batch_size
        output = output.view(
            batch_size,
            k_size,
            output.size(1),
            output.size(2),
            output.size(3),
            output.size(4))
        output = output.view(
            batch_size,
            k_size * output.size(2),
            output.size(3),
            output.size(4),
            output.size(5))  # (b, k * d, t, h, w)

        attention_normalized = self.norm2(attention)
        output = self.mixer(attention_normalized, output_shape=output_shape)

        return output
