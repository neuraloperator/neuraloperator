from functools import partial
import logging
import numpy as np
import torch
import torch
import math
from torch import nn
import torch.nn.functional as F
from .fno_block import FNOBlocks
from .spectral_convolution import SpectralConv

einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

class CODABlocks(nn.Module):
    """Class for the Co-domain Attention Blocks (CODABlocks).

    Parameters
    ----------
    n_modes : list
        Number of modes for each dimension used in K, Q, V operator.
    n_heads : int
        Number of heads for the attention mechanism.
    token_codimension : int
        Co-dimension of each variable or number of channels associated with each variable.
    output_scaling_factor : float
        Scaling factor for the output.
    incremental_n_modes : list
        Incremental number of modes for each dimension (for incremental training).
    head_codimension : int
        Co-dimension of each output token for each head.
    use_channel_mlp : bool, optional
        Whether to use MLP layers to parameterize skip connections. Default is False.
    non_linearity : callable
        Non-linearity function to be used.
    preactivation : bool, optional
        Whether to use preactivation. Default is False.
    fno_skip : str, optional
        Type of skip connection to be used. Default is 'linear'.
    mlp_skip : str, optional
        Module to use for ChannelMLP skip connections. Default is "soft-gating".
    channel_mlp_expansion : float, optional
        Expansion parameter for self.channel_mlp. Default is 0.5.
    separable : bool, optional
        Whether to use separable convolutions. Default is False.
    factorization : str, optional
        Type of factorization to be used. Default is 'tucker'.
    rank : float, optional
        Rank of the factorization. Default is 1.0.
    spectral_convolution : callable
        Spectral convolution module to be used.
    joint_factorization : bool, optional
        Whether to factorize all spectralConv weights as one tensor. Default is False.
    Normalizer : callable
        Normalization module to be used.
    codimension_size : int
        Size of the codimension for the whole function. Only used for permutation_eq = False.
    per_channel_attention : bool, optional
        Whether to use per-channel attention. Default is True (overwrites token_codimension to 1).
    permutation_eq : bool, optional
        Whether to use permutation equivariant mixer layer after the attention mechanism.
    temperature : float
        Temperature parameter for the attention mechanism.
    nonlinear_attention : bool, optional
        Whether to use non-linear activation for K, Q, V operator.
    scale : int
        Scale for downsampling Q, K functions before calculating the attention matrix.
        Higher scale will downsample more.
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
        channel_mlp_skip='linear',
        channel_mlp_expansion=1.0,
        separable=False,
        factorization='tucker',
        rank=1.0,
        spectral_convolution=None,
        norm="instance_norm",
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation='factorized',
        decomposition_kwargs=None,
        fft_norm='forward',
        codimension_size=None,
        per_channel_attention=True,
        permutation_eq=True,
        temperature=1.0,
        nonlinear_attention=False,
        scale=None,
        **_kwargs,
    ):
        super().__init__()

        # Co-dimension of each variable/token. The token embedding space is
        # identical to the variable space, so their dimensionalities are equal.
        self.token_codimension = token_codimension

        # codim of attention from each head
        self.head_codimension = (head_codimension
                                 if head_codimension is not None
                                 else token_codimension)
        self.n_heads = n_heads  # number of heads
        self.output_scaling_factor = output_scaling_factor
        self.temperature = temperature
        self.num_dims = len(n_modes)

        if norm is None:
            Normalizer = torch.nn.Identity
        elif norm == "instance_norm":
            Normalizer = partial(
                nn.InstanceNorm2d,
                affine=True) if self.num_dims == 2 else partial(
                nn.InstanceNorm3d,
                affine=True)
        else:
            raise ValueError(f"Unknown normalization type {norm}")

        if spectral_convolution is None:
            spectral_convolution = SpectralConv

        # K,Q,V operator with or without non_liniarity
        if nonlinear_attention:
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
        if scale is None:
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
        self.Key = FNOBlocks(
            output_scaling_factor=1 / scale,
            conv_module=spectral_convolution,
            **kqv_args,
            **common_args,
        )
        self.Query = FNOBlocks(
            output_scaling_factor=1 / scale,
            conv_module=spectral_convolution,
            **kqv_args,
            **common_args,
        )
        self.Value = FNOBlocks(
            output_scaling_factor=1,
            conv_module=spectral_convolution,
            **kqv_args,
            **common_args,
        )

        if self.n_heads * self.head_codimension != self.token_codimension:
            self.multi_head_proj = FNOBlocks(
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
            self.multi_head_proj = None

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

    def compute_attention(self, tokens, batch_size):
        """
        Compute the key-query-value variant of the attention matrix for input token functions.

        Parameters
        ----------
        tokens : torch.Tensor
            Input tokens with shape (b * t, d, h, w, ...), where:
            b is the batch size,
            t is the number of tokens,
            d is the token codimension,
            and h, w, ... are the domain dimensions.
            Assumes input tokens have been normalized.
        
        batch_size : int
            The size of the batch.
        """

        k = self.Key(tokens)
        q = self.Query(tokens)
        v = self.Value(tokens)
        assert k.size(
            1) % self.n_heads == 0, "Number of channels in k, q, and v should be divisible by number of heads"

        # reshape from (b*t) (n*d) h w -> b n t (d*h*w ...)
        t = k.size(0) // batch_size  # Compute the number of tokens `t`
        # Computer per head token codimension `d`
        d = k.size(1) // self.n_heads

        # reshape from (b*t) (n*d) h w ... to b n t d h w ...
        k = k.view(batch_size, t, self.n_heads, d, *k.shape[-self.num_dims:])
        q = q.view(batch_size, t, self.n_heads, d, *q.shape[-self.num_dims:])
        v = v.view(batch_size, t, self.n_heads, d, *v.shape[-self.num_dims:])

        k = torch.transpose(k, 1, 2)
        q = torch.transpose(q, 1, 2)
        v = torch.transpose(v, 1, 2)
        # resahpe
        k = k.view(batch_size, self.n_heads, t, -1)
        q = q.view(batch_size, self.n_heads, t, -1)
        v = v.view(batch_size, self.n_heads, t, -1)

        # attention mechanism
        dprod = (torch.matmul(q, k.transpose(-1, -2)) /
                 (np.sqrt(k.shape[-1]) * self.temperature))
        dprod = F.softmax(dprod, dim=-1)

        attention = torch.matmul(dprod, v)

        # Reshape from (b, n, t, d * h * w...) to ((b*t), (n*d), h, w, ...)
        attention = attention.transpose(1, 2)
        attention = attention.view(attention.size(0) * attention.size(1),
                                      attention.size(2) * d,
                                      *tokens.shape[-self.num_dims:])
        
        return attention

    def forward(self, x):
        if self.permutation_eq:
            return self._forward_equivariant(x)
        else:
            return self._forward_non_equivariant(x)

    def _forward_equivariant(self, x):
        """Use permutation equivariant mixer layer after the attention mechanism. 
        Shares the same mixer layer for all the variables.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (b, t * d, h, w, ...), where:
            b is the batch size,
            t is the number of tokens,
            and d is the token codimension.
        """
        batch_size = x.shape[0]
        output_shape = x.shape[-self.num_dims:]

        assert x.shape[1] % self.token_codimension == 0, "Number of channels in x should be divisible by token_codimension"

        # reshape from shape b (t*d) h w ... to (b*t) d h w ...
        t = x.size(1) // self.token_codimension
        tokens = x.view(
            x.size(0) * t,
            self.token_codimension,
            *x.shape[-self.num_dims:])

        # normalization and attention mechanism
        tokens_norm = self.norm1(tokens)
        attention = self.compute_attention(tokens_norm, batch_size)
        if self.multi_head_proj is not None:
            attention = self.multi_head_proj(attention)
        attention = self.attention_normalizer(attention + tokens)
        attention_normalized = self.norm2(attention)
        output = self.mixer(attention_normalized, output_shape=output_shape)
        output = self.mixer_out_normalizer(output) + attention

        # reshape from shape (b*t) d h w... to b (t d) h w ...
        t = output.size(0) // batch_size
        output = output.view(
            batch_size,
            t * output.size(1),
            *output.shape[-self.num_dims:])

        return output

    def _forward_non_equivariant(self, x):
        """uses non-permuatation equivariant mixer
        layer and normalizations.
        Parameters
        ----------
        x: torch.tensor. 
        Has shape (b, t*d, h, w, ...) 
        where, t = number of tokens, d = token codimension
        """

        batch_size = x.shape[0]
        output_shape = x.shape[-self.num_dims:]

        assert x.shape[1] % self.token_codimension == 0, "Number of channels in x should be divisible by token_codimension"

        # reshape from shape b (t*d) h w ... to (b*t) d h w ...
        t = x.size(1) // self.token_codimension
        # Normalize the input first
        tokens = self.norm1(x)
        tokens = tokens.view(
            x.size(0) * t,
            self.token_codimension,
            *x.shape[-self.num_dims:])

        # apply attention mechanism
        attention = self.compute_attention(tokens, batch_size)
        if self.multi_head_proj is not None:
            attention = self.multi_head_proj(attention)

        attention = self.attention_normalizer(attention + tokens)

        # reshape for shape '(b*t) d h w.." to "b (t*d) h w ...'
        t = attention.size(0) // batch_size
        attention = attention.view(
            batch_size,
            t * attention.size(2),
            *attention.shape[-self.num_dims:])

        attention_normalized = self.norm2(attention)
        output = self.mixer(attention_normalized, output_shape=output_shape)

        output = self.mixer_out_normalizer(output) + attention

        return output
