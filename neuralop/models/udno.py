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
        in_channels: int,
        out_channels: int,
        in_shape: Tuple[int, int],
        radius_cutoff: float,
        hidden_channels: int = 32,
        num_pool_layers: int = 4,
        disco_kernel_shape: int | List[int] = [3, 4],
        disco_kernel_bias: bool = False,
        drop_prob: float = 0.0,
    ):
        super().__init__()

        assert len(in_shape) == 2, "Input shape must be a 2d tuple like Ex: (64, 64)"

        self.in_shape = in_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_pool_layers = num_pool_layers
        self.disco_kernel_shape = disco_kernel_shape
        self.disco_kernel_bias = disco_kernel_bias
        self.radius_cutoff = radius_cutoff
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList(
            [
                DISCOBlock(
                    in_channels,
                    hidden_channels,
                    in_shape,
                    radius_cutoff,
                    drop_prob,
                    disco_kernel_shape,
                    disco_kernel_bias,
                )
            ]
        )
        ch = hidden_channels
        shape = (in_shape[0] // 2, in_shape[1] // 2)
        radius_cutoff = radius_cutoff * 2
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(
                DISCOBlock(
                    ch,
                    ch * 2,
                    in_shape=shape,
                    radius_cutoff=radius_cutoff,
                    drop_prob=drop_prob,
                    kernel_shape=self.disco_kernel_shape,
                    kernel_bias=self.disco_kernel_bias,
                )
            )
            ch *= 2
            shape = (shape[0] // 2, shape[1] // 2)
            radius_cutoff *= 2

            # test commit

        self.bottleneck = DISCOBlock(
            ch,
            ch * 2,
            in_shape=shape,
            radius_cutoff=radius_cutoff,
            drop_prob=drop_prob,
            kernel_shape=self.disco_kernel_shape,
            kernel_bias=self.disco_kernel_bias,
        )

        self.up = nn.ModuleList()
        self.up_transpose = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose.append(
                TransposeDISCOBlock(
                    ch * 2,
                    ch,
                    in_shape=shape,
                    radius_cutoff=self.disco_radius_cutoff,
                    kernel_shape=self.disco_kernel_shape,
                    kernel_bias=self.disco_kernel_bias,
                )
            )
            shape = (shape[0] * 2, shape[1] * 2)
            radius_cutoff /= 2
            self.up.append(
                DISCOBlock(
                    ch * 2,
                    ch,
                    in_shape=shape,
                    radius_cutoff=radius_cutoff,
                    drop_prob=drop_prob,
                    kernel_shape=self.disco_kernel_shape,
                    kernel_bias=self.disco_kernel_bias,
                )
            )
            ch //= 2

        self.up_transpose.append(
            TransposeDISCOBlock(
                ch * 2,
                ch,
                in_shape=shape,
                radius_cutoff=radius_cutoff,
                kernel_shape=self.disco_kernel_shape,
                kernel_bias=self.disco_kernel_bias,
            )
        )
        shape = (shape[0] * 2, shape[1] * 2)
        radius_cutoff /= 2
        self.up.append(
            nn.Sequential(
                DISCOBlock(
                    ch * 2,
                    ch,
                    in_shape=shape,
                    radius_cutoff=radius_cutoff,
                    drop_prob=drop_prob,
                    kernel_shape=self.disco_kernel_shape,
                    kernel_bias=self.disco_kernel_bias,
                ),
                nn.Conv2d(
                    ch, self.out_chans, kernel_size=1, stride=1
                ),  # 1x1 conv is always res-invariant (pixel wise channel transformation)
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        stack = []
        output = image

        # encoder: apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        # bottleneck
        output = self.bottleneck(output)

        # decoder: apply up-sampling layers
        for transpose, disco in zip(self.up_transpose, self.up):
            downsample_layer = stack.pop()
            output = transpose(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = disco(output)

        return output


class DISCOBlock(nn.Module):
    """
    A DISCO Encoder Block that consists of two Equidistant DISCO2d layers each
    followed by instance normalization, LeakyReLU activation and dropout.
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
        in_channels: int,
        out_channels: int,
        in_shape: Tuple[int, int],
        radius_cutoff: float,
        kernel_shape: int | List[int] = [6, 7],
        kernel_bias: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_shape = in_shape
        self.radius_cutoff = radius_cutoff
        self.kernel_shape = kernel_shape
        self.kernel_bias = kernel_bias

        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            EquidistantDiscreteContinuousConv2d(
                self.in_channels,
                self.out_channels,
                in_shape=(2 * self.in_shape[0], 2 * self.in_shape[1]),
                out_shape=self.in_shape,  # TODO: fix
                kernel_shape=self.kernel_shape,
                bias=self.bias,
                radius_cutoff=(self.radius_cutoff / 2),
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)
