from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F

from .base_model import BaseModel
from ..layers.discrete_continuous_convolution import (
    EquidistantDiscreteContinuousConv2d,
)


class UDNO(BaseModel, name="UDNO"):
    """2-Dimensional U-shaped DISCO Neural Operator (UDNO) [1]_. This module is
    meant to be an in-place operator replacement for the U-Net. The model uses
    Equidistant Discrete-continuous convolutions (DISCO) on equidistant/regular
    grids [2]_.

    Parameters
    ----------
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels
    in_shape : tupe[int, int]
        2 dimensional input shape, must be square as DISCO assumes inputs are
        on a regular square [-1, 1]^2 grid
    disco_radius_cutoff : float, optional
            Value in (0, 1). Controls the effective size of the DISCO kernel.
        Compiled kernels have size (SxS) where S = ceil(max(in_shape[0], in_shape[1]) * disco_radius_cutoff)
            DISCO kernels compile to always cover the same relative size of the
        full input. If you are going to train on one resolution and want to
        infer at some other resolution, you must pass in this value.
            For example, with an input shape of (64, 64), a disco_radius_cutoff of
        0.04 would lead to an effective kernel size of 3 = ceil(64 * 0.04). When
        using even effective kernel_sizes, outputs are zero-padded so
        in_shape = out_shape.
    hidden_channels : int, optional
        Number of output channels of first DISCO layer. 32 by default.
    num_pool_layers : int, optional
        Number of down-sampling and up-sampling layers. 4 by default.
    disco_kernel_shape : int | list[int], optional
        Shape of the DISCO kernel.
        provide two integers for anisotropic kernels: [# rings, # basis functions]
        provide a single integer for isotropic kernels
        [3, 4] by default: 3 rings, 4 basis functions

    References
    ----------
    .. [1] Jatyani, A. S., Wang, J., Wu, Z., Liu-Schiaffini, M., Tolooshams, B., Anandkumar, A.;
        Unifying Subsampling Pattern Variations for Compressed Sensing MRI with Neural Operators;
        arxiv:2410.16290, 2024.

    .. [2] Liu-Schiaffini M., Berner J., Bonev B., Kurth T., Azizzadenesheli K., Anandkumar A.;
        Neural Operators with Localized Integral and Differential Kernels;  arxiv:2402.16845
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: tuple[int, int],
        disco_radius_cutoff: Optional[float],
        hidden_channels: int = 32,
        num_pool_layers: int = 4,
        disco_kernel_shape: int | list[int] = [3, 4],
        disco_kernel_bias: bool = False,
        drop_prob: float = 0.0,
    ):
        super().__init__()

        assert (
            len(in_shape) == 2
        ), "[model:UDNO:init] Input shape must be a 2d tuple like Ex: (64, 64)"

        assert (
            in_shape[0] == in_shape[1]
        ), "[model:UDNO:init] Input shape must be square. DISCO kernels assume inputs are on a [-1, 1] square grid."

        if disco_radius_cutoff is None:
            disco_radius_cutoff = 3 / min(in_shape) / 2

        self.in_shape = in_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_pool_layers = num_pool_layers
        self.disco_kernel_shape = disco_kernel_shape
        self.disco_kernel_bias = disco_kernel_bias
        self.disco_radius_cutoff = disco_radius_cutoff
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList(
            [
                DISCOBlock(
                    in_channels,
                    hidden_channels,
                    in_shape,
                    disco_radius_cutoff,
                    drop_prob,
                    disco_kernel_shape,
                    disco_kernel_bias,
                )
            ]
        )
        ch = hidden_channels
        shape = (in_shape[0] // 2, in_shape[1] // 2)
        disco_radius_cutoff = disco_radius_cutoff * 2
        for _ in range(num_pool_layers - 1):
            if 0 in shape:
                raise ValueError(
                    "[models:UDNO:init] The number of pool layers is too many. In each downsampling part of the U-shaped architecture, the spatial dimensions halve. If there are too many downsampling layers (number of pools is too large), the spatial dimensions go to 0. Please pass in a lower number of pool layers and try again."
                )

            self.down_sample_layers.append(
                DISCOBlock(
                    ch,
                    ch * 2,
                    in_shape=shape,
                    radius_cutoff=disco_radius_cutoff,
                    drop_prob=drop_prob,
                    kernel_shape=self.disco_kernel_shape,
                    kernel_bias=self.disco_kernel_bias,
                )
            )
            ch *= 2
            shape = (shape[0] // 2, shape[1] // 2)
            disco_radius_cutoff *= 2

        self.bottleneck = DISCOBlock(
            ch,
            ch * 2,
            in_shape=shape,
            radius_cutoff=disco_radius_cutoff,
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
            disco_radius_cutoff /= 2
            self.up.append(
                DISCOBlock(
                    ch * 2,
                    ch,
                    in_shape=shape,
                    radius_cutoff=disco_radius_cutoff,
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
                radius_cutoff=disco_radius_cutoff,
                kernel_shape=self.disco_kernel_shape,
                kernel_bias=self.disco_kernel_bias,
            )
        )
        shape = (shape[0] * 2, shape[1] * 2)
        disco_radius_cutoff /= 2
        self.up.append(
            nn.Sequential(
                DISCOBlock(
                    ch * 2,
                    ch,
                    in_shape=shape,
                    radius_cutoff=disco_radius_cutoff,
                    drop_prob=drop_prob,
                    kernel_shape=self.disco_kernel_shape,
                    kernel_bias=self.disco_kernel_bias,
                ),
                nn.Conv2d(
                    ch, self.out_channels, kernel_size=1, stride=1
                ),  # 1x1 conv is always res-invariant (pixel wise channel transformation)
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _, _, h, w = image.shape
        assert (
            h == self.in_shape[0] and w == self.in_shape[1]
        ), "[model:UDNO:forward] tensor passed into forward pass must have same shape as `in_shape`: {self.in_shape}"
        assert h == w, "[model:UDNO:forward] tensors passed to UDNO must be square"

        stack: list[torch.Tensor] = []
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

            padding = [0, 0, 0, 0]  # [left, right, top, bottom]

            diff_w = downsample_layer.shape[-1] - output.shape[-1]
            if diff_w > 0:
                padding[0] = diff_w // 2  # padding left
                padding[1] = diff_w - padding[0]  # padding right

            diff_h = downsample_layer.shape[-2] - output.shape[-2]
            if diff_h > 0:
                padding[2] = diff_h // 2  # padding top
                padding[3] = diff_h - padding[2]  # padding bottom

            if any(padding):
                output = F.pad(output, padding, mode="reflect")

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
        in_shape: tuple[int, int],
        radius_cutoff: float,
        drop_prob: float,
        kernel_shape: int | list[int] = [6, 7],
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
                bias=self.kernel_bias,
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
                bias=self.kernel_bias,
                radius_cutoff=self.radius_cutoff,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            _, _, h, w = image.shape
            image = layer(image)

            # if DISCO compiles to an even-sized kernel, we need to zero_pad to preserve spatial dims
            # FIXME: replace with `pad_output` flag when support is added to DISCO
            if isinstance(layer, EquidistantDiscreteContinuousConv2d):
                # check if padding is required, get h_ (h') and w_ (w')
                _, _, h_, w_ = image.shape
                # prev - curr b/c h_ <= h and w_ <= w
                pad_h = h - h_
                pad_w = w - w_
                if pad_h > 0 or pad_w > 0:
                    # pad format is (left, right, top, bottom)
                    pad_left = pad_w // 2
                    pad_right = pad_w - pad_left
                    pad_top = pad_h // 2
                    pad_bottom = pad_h - pad_top
                    image = F.pad(
                        image,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode="constant",
                        value=0,
                    )
        return image


class TransposeDISCOBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: tuple[int, int],
        radius_cutoff: float,
        kernel_shape: int | list[int] = [6, 7],
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
                out_shape=(
                    2 * self.in_shape[0],
                    2 * self.in_shape[1],
                ),
                kernel_shape=self.kernel_shape,
                bias=self.kernel_bias,
                radius_cutoff=(self.radius_cutoff / 2),
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)
