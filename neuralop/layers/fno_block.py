from typing import List, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from .channel_mlp import ChannelMLP
from .complex import CGELU, apply_complex, ctanh, ComplexValued
from .normalization_layers import AdaIN, InstanceNorm
from .skip_connections import skip_connection
from .spectral_convolution import SpectralConv
from ..utils import validate_scaling_factor


Number = Union[int, float]


class FNOBlocks(nn.Module):
    """FNOBlocks implements a sequence of Fourier layers
    as described in "Fourier Neural Operator for Parametric
    Partial Differential Equations (Li et al., 2021).
    Parameters
    ----------
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
        output_scaling_factor : Optional[Union[Number, List[Number]]], optional
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
            dropout parameter for self.channel_mlp, by default 0
        channel_mlp_expansion : float, optional
            expansion parameter for self.channel_mlp, by default 0.5
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
            module to use for ChannelMLP skip connections, by default "soft-gating"
            see layers.skip_connections for more details
        complex_spatial_data : bool, optional
            whether the FNO's data takes on complex values in space, by default False
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
        joint_factorization : bool, optional
            whether to factorize all spectralConv weights as one tensor, by default False
        fixed_rank_modes : bool, optional
            fixed_rank_modes parameter for SpectralConv, by default False
        implementation : str, optional
            implementation parameter for SpectralConv, by default "factorized"
        decomposition_kwargs : _type_, optional
            kwargs for tensor decomposition in SpectralConv, by default dict()
        fft_norm : str, optional
            how to normalize discrete fast Fourier transform, by default "forward"
            if "forward", normalize just the forward direction F(v(x)) by 1/n (number of total modes)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        output_scaling_factor=None,
        n_layers=1,
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
        complex_spatial_data=False,
        separable=False,
        factorization=None,
        rank=1.0,
        SpectralConv=SpectralConv,
        joint_factorization=False,
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
        self.n_dim = len(n_modes)

        self.output_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(output_scaling_factor, self.n_dim, n_layers)

        self.max_n_modes = max_n_modes
        self.fno_block_precision = fno_block_precision
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.stabilizer = stabilizer
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = fno_skip
        self.channel_mlp_skip = channel_mlp_skip
        self.complex_spatial_data = complex_spatial_data
        self.use_channel_mlp = use_channel_mlp
        self.channel_mlp_expansion = channel_mlp_expansion
        self.channel_mlp_dropout = channel_mlp_dropout
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features

        # apply real nonlin if data is real, otherwise CGELU
        if complex_spatial_data:
            self.non_linearity = CGELU
        else:
            self.non_linearity = non_linearity

        self.convs = SpectralConv(
            self.in_channels,
            self.out_channels,
            self.n_modes,
            output_scaling_factor=output_scaling_factor,
            max_n_modes=max_n_modes,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            n_layers=n_layers,
            complex_spatial_data=complex_spatial_data
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
        if self.complex_spatial_data:
            self.fno_skips = nn.ModuleList(
                [ComplexValued(x) for x in self.fno_skips]
                )

        if use_channel_mlp:
            self.channel_mlp = nn.ModuleList(
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
            if self.complex_spatial_data:
                self.channel_mlp = nn.ModuleList(
                    [ComplexValued(x) for x in self.channel_mlp]
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
            if self.complex_spatial_data:
                self.channel_mlp_skips = nn.ModuleList(
                    [ComplexValued(x) for x in self.channel_mlp_skips]
                )
        else:
            self.channel_mlp = None

        # Each block will have 2 norms if we also use a ChannelMLP
        self.n_norms = 1 if self.channel_mlp is None else 2
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
        # elif norm == 'layer_norm':
        #     self.norm = nn.ModuleList(
        #         [
        #             nn.LayerNorm(elementwise_affine=False)
        #             for _ in range(n_layers*self.n_norms)
        #         ]
        #     )
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
        if self.preactivation:
            return self.forward_with_preactivation(x, index, output_shape)
        else:
            return self.forward_with_postactivation(x, index, output_shape)

    def forward_with_postactivation(self, x, index=0, output_shape=None):
        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.channel_mlp is not None:
            x_skip_channel_mlp = self.channel_mlp_skips[index](x)
            x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            if self.complex_spatial_data:
                x = ctanh(x)
            else:
                x = torch.tanh(x)

        x_fno = self.convs(x, index, output_shape=output_shape)

        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](x_fno)

        x = x_fno + x_skip_fno

        if (self.channel_mlp is not None) or (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        if self.channel_mlp is not None:
            x = self.channel_mlp[index](x) + x_skip_channel_mlp

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

        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.channel_mlp is not None:
            x_skip_channel_mlp = self.channel_mlp_skips[index](x)
            x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            if self.complex_spatial_data:
                x = ctanh(x)
            else:
                x = torch.tanh(x)

        x_fno = self.convs(x, index, output_shape=output_shape)
        x = x_fno + x_skip_fno

        if self.channel_mlp is not None:
            if index < (self.n_layers - 1):
                x = self.non_linearity(x)

            if self.norm is not None:
                x = self.norm[self.n_norms * index + 1](x)

            x = self.channel_mlp[index](x) + x_skip_channel_mlp

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.convs.n_modes = n_modes
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


class SubModule(nn.Module):
    """Class representing one of the sub_module from the mother joint module

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules,
    they all point to the same data, which is shared.
    """

    def __init__(self, main_module, indices):
        super().__init__()
        self.main_module = main_module
        self.indices = indices

    def forward(self, x):
        return self.main_module.forward(x, self.indices)