# types
from typing import List, Optional, Tuple

# internal dependencies
from .base_model import BaseModel
from ..layers.discrete_continuous_convolution import (
    EquidistantDiscreteContinuousConv2d,
    EquidistantDiscreteContinuousConvTranspose2d,
)

import torch
import torch.nn as nn
from torch.nn import functional as F


class UDNO(BaseModel, name="UDNO"):
    def __init__(
        self,
        in_shape: Tuple[int, int],
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 32,
        num_pool_layers: int = 4,
        disco_kernel_shape: Tuple[int, int] = (3, 4),
        radius_cutoff: Optional[float] = None,
        drop_prob: float = 0.0,
    ):
        super().__init__()

        assert len(in_shape) == 2, "Input shape must be 2D"

        self.in_shape = in_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_pool_layers = num_pool_layers
        self.disco_kernel_shape = disco_kernel_shape
        self.radius_cutoff = radius_cutoff
        self.drop_prob = drop_prob


class DISCOBlock(nn.Module):
    """
    A DISCO Block that consists of two DISCO layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: Tuple[int, int],
        radius_cutoff: float,
        drop_prob: float,
        kernel_shape: int | List[int] = [6, 7],
        kernel_bias: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_shape = in_shape
        self.radius_cutoff = radius_cutoff
        self.drop_prob = drop_prob
        self.kernel_shape = kernel_shape
        self.kernel_bias = kernel_bias

        self.layers = nn.Sequential(
            EquidistantDiscreteContinuousConv2d(
                self.in_channels,
                self.out_channels,
                self.in_shape,
                self.in_shape,
                self.kernel_shape,
                bias=self.bias,
                radius_cutoff=self.radius_cutoff,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            EquidistantDiscreteContinuousConv2d(
                self.out_channels,
                self.out_channels,
                self.in_shape,
                self.in_shape,
                self.kernel_shape,
                bias=self.bias,
                radius_cutoff=self.radius_cutoff,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class TransposeDISCOBlock(nn.Module):

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        radius_cutoff: float,
        in_shape: Tuple[int, int],
        kernel_shape: Tuple[int, int] = (3, 4),
    ):
        """
        Parameters
        ----------
        in_chans : int
            Number of channels in the input.
        out_chans : int
            Number of channels in the output.
        radius_cutoff : float
            Control the effective radius of the DISCO kernel. Values are
            between 0.0 and 1.0. The radius_cutoff is represented as a proportion
            of the normalized input space, to ensure that kernels are resolution
            invaraint.
        in_shape : Tuple[int]
            Unbatched spatial 2D shape of the input to this block.
            Rrequired to dynamically compile DISCO kernels for resolution invariance.
        kernel_shape : Tuple[int, int], optional
            Shape of the DISCO kernel. Default is (3, 4). This corresponds to 3
            rings and 4 anisotropic basis functions. Under the hood, each DISCO
            kernel has (3 - 1) * 4 + 1 = 9 parameters, equivalent to a standard
            3x3 convolution kernel.

            Note: This is NOT kernel_size, as under the DISCO framework,
            kernels are dynamically compiled to support resolution invariance
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DISCO2d(
                in_chans,
                out_chans,
                kernel_shape=kernel_shape,
                in_shape=(2 * in_shape[0], 2 * in_shape[1]),
                bias=False,
                radius_cutoff=(radius_cutoff / 2),
                padding_mode="constant",
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image : torch.Tensor
            Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
