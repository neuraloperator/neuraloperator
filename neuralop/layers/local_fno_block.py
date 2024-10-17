from typing import List, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from .channel_mlp import ChannelMLP
from .fno_block import SubModule
from .differential_conv import FiniteDifferenceConvolution
from .normalization_layers import AdaIN, InstanceNorm
from .skip_connections import skip_connection
from .spectral_convolution import SpectralConv
from ..utils import validate_scaling_factor


Number = Union[int, float]


class LocalFNOBlocks(nn.Module):
    """LocalFNOBlocks implements a sequence of Fourier layers
    as described in "Fourier Neural Operator for Parametric
    Partial Differential Equations (Li et al., 2021).

    The Fourier layers are placed in parallel with differential 
    kernel layers from "Neural Operators with Localized Integral 
    and Differential Kernels" (Liu-Schiaffini et al., 2024).
    
    Parameters
        ----------
        in_channels : int
            input channels to Fourier layers
        out_channels : int
            output channels after Fourier layers
        n_modes : int, List[int]
            number of modes to keep along each dimension 
            in frequency space. Can either be specified as
            an int (for all dimensions) or an iterable with one
            number per dimension
        resolution_scaling_factor : Optional[Union[Number, List[Number]]], optional
            factor by which to scale outputs for super-resolution, by default None
        n_layers : int, optional
            number of Fourier layers to apply in sequence, by default 1
        max_n_modes : int, List[int], optional
            maximum number of modes to keep along each dimension, by default None
        fno_block_precision : str, optional
            floating point precision to use for computations, by default "full"
        use_channel_mlp : bool, optional
            whether to use mlp layers to parameterize skip connections, by default False
        channel_mlp_dropout : int, optional
            dropout parameter for self.mlp, by default 0
        channel_mlp_expansion : float, optional
            expansion parameter for self.mlp, by default 0.5
        non_linearity : torch.nn.F module, optional
            nonlinear activation function to use between layers, by default F.gelu
        stabilizer : Literal["tanh"], optional
            stabilizing module to use between certain layers, by default None
            if "tanh", use tanh
        norm : Literal["ada_in", "group_norm", "instance_norm"], optional
            Normalization layer to use, by default None
        ada_in_features : int, optional
            number of features for adaptive instance norm above, by default None
        preactivation : bool, optional
            whether to call forward pass with pre-activation, by default False
            if True, call nonlinear activation and norm before Fourier convolution
            if False, call activation and norms after Fourier convolutions
        fno_skip : str, optional
            module to use for FNO skip connections, by default "linear"
            see layers.skip_connections for more details
        channel_mlp_skip : str, optional
            module to use for MLP skip connections, by default "soft-gating"
            see layers.skip_connections for more details
        SpectralConv Params
        -------------------
        separable : bool, optional
            separable parameter for SpectralConv, by default False
        factorization : str, optional
            factorization parameter for SpectralConv, by default None
        rank : float, optional
            rank parameter for SpectralConv, by default 1.0
        SpectralConv : BaseConv, optional
            module to use for SpectralConv, by default SpectralConv
        fixed_rank_modes : bool, optional
            fixed_rank_modes parameter for SpectralConv, by default False
        implementation : str, optional
            implementation parameter for SpectralConv, by default "factorized"
        decomposition_kwargs : _type_, optional
            kwargs for tensor decomposition in SpectralConv, by default dict()
        fft_norm : str, optional
            how to normalize discrete fast Fourier transform, by default "forward"
            if "forward", normalize just the forward direction F(v(x)) by 1/n (number of total modes)
        FiniteDifferenceConvolution Params
        ----------------------------------
        diff_layers : bool list, optional
            Must be same length as n_layers, dictates whether to include a
            differential kernel parallel connection at each layer
        fin_diff_implementation : str in ['subtract_middle', 'subtract_all'], optional
            Implementation type for FiniteDifferenceConvolution.
            See differential_conv.py.
        conv_padding_mode : str in ['periodic', 'circular', 'replicate', 'reflect', 'zeros'], optional
            Padding mode for spatial convolution kernels.
        default_grid_res : int or None, optional
            Proportional to default input shape of last spatial dimension. If 
            None, inferred from data. This is used for defining the appropriate
            scaling of the differential kernel.
        fin_diff_kernel_size : odd int, optional
            Conv kernel size for finite difference convolution.
        mix_derivatives : bool, optional
            Whether to mix derivatives across channels
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        resolution_scaling_factor=None,
        n_layers=1,
        diff_layers=[True],
        fin_diff_implementation='subtract_middle',
        conv_padding_mode='periodic',
        default_grid_res=None,
        fin_diff_kernel_size=3,
        mix_derivatives=True,
        max_n_modes=None,
        fno_block_precision="full",
        use_channel_mlp=False,
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        ada_in_features=None,
        preactivation=False,
        fno_skip="linear",
        channel_mlp_skip="soft-gating",
        separable=False,
        factorization=None,
        rank=1.0,
        conv_module=SpectralConv,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        fft_norm="forward",
        **kwargs,
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes

        if len(n_modes) > 3 and True in diff_layers:
            NotImplementedError("Differential convs not implemented for dimensions higher than 3.")
            
        self.n_dim = len(n_modes)

        self.resolution_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(resolution_scaling_factor, self.n_dim, n_layers)

        self.max_n_modes = max_n_modes
        self.fno_block_precision = fno_block_precision
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.non_linearity = non_linearity
        self.stabilizer = stabilizer
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = fno_skip
        self.channel_mlp_skip = channel_mlp_skip
        self.use_channel_mlp = use_channel_mlp
        self.channel_mlp_expansion = channel_mlp_expansion
        self.channel_mlp_dropout = channel_mlp_dropout
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features

        self.diff_layers = diff_layers
        self.fin_diff_implementation = fin_diff_implementation
        self.conv_padding_mode = conv_padding_mode
        self.default_grid_res = default_grid_res
        self.fin_diff_kernel_size = fin_diff_kernel_size
        self.mix_derivatives = mix_derivatives

        assert len(diff_layers) == n_layers, "Length of diff_layers must be n_layers"

        self.convs = nn.ModuleList(
            [
                conv_module(
                    self.in_channels,
                    self.out_channels,
                    self.n_modes,
                    resolution_scaling_factor=None if resolution_scaling_factor is None else self.resolution_scaling_factor[i],
                    max_n_modes=max_n_modes,
                    rank=rank,
                    fixed_rank_modes=fixed_rank_modes,
                    implementation=implementation,
                    separable=separable,
                    factorization=factorization,
                    decomposition_kwargs=decomposition_kwargs,
                ) for i in range(n_layers)
            ]
        )

        self.fno_skips = nn.ModuleList(
            [
                skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type=fno_skip,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )

        self.groups = 1 if mix_derivatives else in_channels
        self.differential = nn.ModuleList(
            [
                FiniteDifferenceConvolution(self.in_channels, self.out_channels,
                                            self.n_dim, self.fin_diff_kernel_size, 
                                            self.groups, self.conv_padding_mode, fin_diff_implementation)
                for _ in range(sum(self.diff_layers))
            ]
        )

        # Helper for calling differential layers
        self.differential_idx_list = []
        j = 0
        for i in range(n_layers):
            if self.diff_layers[i]:
                self.differential_idx_list.append(j)
                j += 1
            else:
                self.differential_idx_list.append(-1)

        assert max(self.differential_idx_list) == sum(self.diff_layers) - 1

        if use_channel_mlp:
            self.mlp = nn.ModuleList(
                [
                    ChannelMLP(
                        in_channels=self.out_channels,
                        hidden_channels=round(self.out_channels * channel_mlp_expansion),
                        dropout=channel_mlp_dropout,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.channel_mlp_skips = nn.ModuleList(
                [
                    skip_connection(
                        self.in_channels,
                        self.out_channels,
                        skip_type=channel_mlp_skip,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.mlp = None

        # Each block will have 2 norms if we also use an MLP
        self.n_norms = 1 if self.mlp is None else 2
        if norm is None:
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.ModuleList(
                    [
                        InstanceNorm()
                        for _ in range(n_layers * self.n_norms)
                    ]
                )
        elif norm == "group_norm":
            self.norm = nn.ModuleList(
                [
                    nn.GroupNorm(num_groups=1, num_channels=self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        elif norm == "ada_in":
            self.norm = nn.ModuleList(
                [
                    AdaIN(ada_in_features, out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        else:
            raise ValueError(
                f"Got norm={norm} but expected None or one of "
                "[instance_norm, group_norm, ada_in]"
            )

    def set_ada_in_embeddings(self, *embeddings):
        """Sets the embeddings of each Ada-IN norm layers

        Parameters
        ----------
        embeddings : tensor or list of tensor
            if a single embedding is given, it will be used for each norm layer
            otherwise, each embedding will be used for the corresponding norm layer
        """
        if len(embeddings) == 1:
            for norm in self.norm:
                norm.set_embedding(embeddings[0])
        else:
            for norm, embedding in zip(self.norm, embeddings):
                norm.set_embedding(embedding)

    def forward(self, x, index=0, output_shape=None):
        if self.default_grid_res is None:
            self.default_grid_res = x.shape[-1]

        if self.preactivation:
            return self.forward_with_preactivation(x, index, output_shape)
        else:
            return self.forward_with_postactivation(x, index, output_shape)

    def forward_with_postactivation(self, x, index=0, output_shape=None):
        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.mlp is not None:
            x_skip_mlp = self.channel_mlp_skips[index](x)
            x_skip_mlp = self.convs[index].transform(x_skip_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            x = torch.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)

        if self.differential_idx_list[index] != -1:
            grid_width_scaling_factor = 1 / (x.shape[-1] / self.default_grid_res)
            x_differential = self.differential[self.differential_idx_list[index]](x, grid_width_scaling_factor)
            x_differential = self.convs[index].transform(x_differential, output_shape=output_shape)
        else:
            x_differential = 0

        x_fno_diff = x_fno + x_differential

        if self.norm is not None:
            x_fno_diff = self.norm[self.n_norms * index](x_fno_diff)

        x = x_fno_diff + x_skip_fno

        if (self.mlp is not None) or (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        if self.mlp is not None:
            x = self.mlp[index](x) + x_skip_mlp

            if self.norm is not None:
                x = self.norm[self.n_norms * index + 1](x)

            if index < (self.n_layers - 1):
                x = self.non_linearity(x)

        return x

    def forward_with_preactivation(self, x, index=0, output_shape=None):
        # Apply non-linear activation (and norm)
        # before this block's convolution/forward pass:
        x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index](x)

        if self.differential_idx_list[index] != -1:
            grid_width_scaling_factor = 1 / (x.shape[-1] / self.default_grid_res)
            x_differential = self.differential[self.differential_idx_list[index]](x, grid_width_scaling_factor)
        else:
            x_differential = 0

        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno_diff = self.convs[index].transform(x_skip_fno + x_differential, output_shape=output_shape)

        if self.mlp is not None:
            x_skip_mlp = self.channel_mlp_skips[index](x)
            x_skip_mlp = self.convs[index].transform(x_skip_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            x = torch.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)

        x = x_fno + x_skip_fno_diff

        if self.mlp is not None:
            if index < (self.n_layers - 1):
                x = self.non_linearity(x)

            if self.norm is not None:
                x = self.norm[self.n_norms * index + 1](x)

            x = self.mlp[index](x) + x_skip_mlp

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        for i in range(self.n_layers):
            self.convs[i].n_modes = n_modes
        self._n_modes = n_modes

    def get_block(self, indices):
        """Returns a sub-FNO Block layer from the jointly parametrized main block

        The parametrization of an FNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single layer is parametrized, directly use the main class."
            )

        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)