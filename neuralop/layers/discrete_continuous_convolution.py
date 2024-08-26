import math
import numpy

import torch
import torch.nn as nn

from typing import Union, List, Optional, Tuple
from functools import partial

# import the base class from torch-harmonics
from torch_harmonics.quadrature import _precompute_grid
from torch_harmonics.convolution import _compute_support_vals_isotropic, _compute_support_vals_anisotropic
from torch_harmonics.convolution import DiscreteContinuousConv


def _normalize_convolution_tensor_2d(psi_idx, psi_vals, grid_in, grid_out, kernel_shape, quad_weights, transpose_normalization=False, eps=1e-9):
    """
    Discretely normalizes the convolution tensor.
    """

    n_in = grid_in.shape[-1]
    n_out = grid_out.shape[-2]

    if len(kernel_shape) == 1:
        kernel_size = math.ceil(kernel_shape[0] / 2)
    elif len(kernel_shape) == 2:
        kernel_size = (kernel_shape[0] // 2) * kernel_shape[1] + kernel_shape[0] % 2

    # # reshape the indices implicitly to be ikernel, n_in, n_out
    # idx = torch.stack([psi_idx[0], psi_idx[1], psi_idx[2] // nlon_in, psi_idx[2] % nlon_in], dim=0)
    idx = psi_idx

    if transpose_normalization:
        # pre-compute the quadrature weights
        q = quad_weights[idx[1]].reshape(-1)

        # loop through dimensions which require normalization
        for ik in range(kernel_size):
            for iin in range(n_in):
                # get relevant entries
                iidx = torch.argwhere((idx[0] == ik) & (idx[2] == iin))
                # normalize, while summing also over the input longitude dimension here as this is not available for the output
                vnorm = torch.sum(psi_vals[iidx] * q[iidx])
                psi_vals[iidx] = psi_vals[iidx] / (vnorm + eps)
    else:
        # pre-compute the quadrature weights
        q = quad_weights[idx[2]].reshape(-1)

        # loop through dimensions which require normalization
        for ik in range(kernel_size):
            for iout in range(n_out):
                # get relevant entries
                iidx = torch.argwhere((idx[0] == ik) & (idx[1] == iout))
                # normalize
                vnorm = torch.sum(psi_vals[iidx] * q[iidx])
                psi_vals[iidx] = psi_vals[iidx] / (vnorm + eps)

    return psi_vals


def _precompute_convolution_tensor_2d(grid_in, grid_out, kernel_shape, quad_weights, radius_cutoff=0.01, periodic=False, transpose_normalization=False):
    """
    Precomputes the translated filters at positions $T^{-1}_j \omega_i = T^{-1}_j T_i \nu$. Similar to the S2 routine,
    only that it assumes a non-periodic subset of the euclidean plane
    """

    # check that input arrays are valid point clouds in 2D
    assert len(grid_in) == 2
    assert len(grid_out) == 2
    assert grid_in.shape[0] == 2
    assert grid_out.shape[0] == 2

    n_in = grid_in.shape[-1]
    n_out = grid_out.shape[-1]

    if len(kernel_shape) == 1:
        kernel_handle = partial(_compute_support_vals_isotropic, nr=kernel_shape[0], r_cutoff=radius_cutoff)
    elif len(kernel_shape) == 2:
        kernel_handle = partial(_compute_support_vals_anisotropic, nr=kernel_shape[0], nphi=kernel_shape[1], r_cutoff=radius_cutoff)
    else:
        raise ValueError("kernel_shape should be either one- or two-dimensional.")

    grid_in = grid_in.reshape(2, 1, n_in)
    grid_out = grid_out.reshape(2, n_out, 1)

    diffs = grid_in - grid_out
    if periodic:
        periodic_diffs = torch.where(diffs > 0.0, diffs - 1, diffs + 1)
        diffs = torch.where(diffs.abs() < periodic_diffs.abs(), diffs, periodic_diffs)

    r = torch.sqrt(diffs[0] ** 2 + diffs[1] ** 2)
    phi = torch.arctan2(diffs[1], diffs[0]) + torch.pi

    idx, vals = kernel_handle(r, phi)

    idx = idx.permute(1, 0)

    vals = _normalize_convolution_tensor_2d(idx, vals, grid_in, grid_out, kernel_shape, quad_weights, transpose_normalization=transpose_normalization)

    return idx, vals


class DiscreteContinuousConv2d(DiscreteContinuousConv):
    """
    Discrete-continuous convolutions (DISCO) [1] on arbitrary 2d grids as implemented for [2].

    [1] Ocampo J., Price M.A. , McEwen J.D.; Scalable and equivariant spherical CNNs by discrete-continuous (DISCO) convolutions, ICLR (2023), arXiv:2209.13603
    [2] Liu-Schiaffini M., Berner J., Bonev B., Kurth T., Azizzadenesheli K., Anandkumar A.; Neural Operators with Localized Integral and Differential Kernels;  arxiv:2402.16845
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_in: torch.Tensor,
        grid_out: torch.Tensor,
        kernel_shape: Union[int, List[int]],
        n_in: Optional[Tuple[int]] = None,
        n_out: Optional[Tuple[int]] = None,
        quad_weights: Optional[torch.Tensor] = None,
        periodic: Optional[bool] = False,
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        radius_cutoff: Optional[float] = None,
    ):
        super().__init__(in_channels, out_channels, kernel_shape, groups, bias)

        # the instantiator supports convenience constructors for the input and output grids
        if isinstance(grid_in, torch.Tensor):
            assert isinstance(quad_weights, torch.Tensor)
            assert not periodic
        elif isinstance(grid_in, str):
            assert n_in is not None
            assert len(n_in) == 2
            x, wx = _precompute_grid(n_in[0], grid=grid_in, periodic=periodic)
            y, wy = _precompute_grid(n_in[1], grid=grid_in, periodic=periodic)
            x, y = torch.meshgrid(torch.from_numpy(x), torch.from_numpy(y))
            wx, wy = torch.meshgrid(torch.from_numpy(wx), torch.from_numpy(wy))
            grid_in = torch.stack([x.reshape(-1), y.reshape(-1)])
            quad_weights = (wx * wy).reshape(-1)
        else:
            raise ValueError(f"Unknown grid input type of type {type(grid_in)}")

        if isinstance(grid_out, torch.Tensor):
            pass
        elif isinstance(grid_out, str):
            assert n_out is not None
            assert len(n_out) == 2
            x, wx = _precompute_grid(n_out[0], grid=grid_out, periodic=periodic)
            y, wy = _precompute_grid(n_out[1], grid=grid_out, periodic=periodic)
            x, y = torch.meshgrid(torch.from_numpy(x), torch.from_numpy(y))
            grid_out = torch.stack([x.reshape(-1), y.reshape(-1)])
        else:
            raise ValueError(f"Unknown grid output type of type {type(grid_out)}")

        # check that input arrays are valid point clouds in 2D
        assert len(grid_in.shape) == 2
        assert len(grid_out.shape) == 2
        assert len(quad_weights.shape) == 1
        assert grid_in.shape[0] == 2
        assert grid_out.shape[0] == 2

        self.n_in = grid_in.shape[-1]
        self.n_out = grid_out.shape[-1]

        # compute the cutoff radius based on the bandlimit of the input field
        # TODO: Attention - this heuristic is ad-hoc! Make sure to set it yourself!
        if radius_cutoff is None:
            radius_cutoff = radius_cutoff = 2 / float(math.sqrt(self.n_out) - 1)

        if radius_cutoff <= 0.0:
            raise ValueError("Error, radius_cutoff has to be positive.")

        # integration weights
        self.register_buffer("quad_weights", quad_weights, persistent=False)

        idx, vals = _precompute_convolution_tensor_2d(grid_in, grid_out, self.kernel_shape, quad_weights, radius_cutoff=radius_cutoff, periodic=periodic)

        # to improve performance, we make psi a matrix by merging the first two dimensions
        # This has to be accounted for in the forward pass
        idx = torch.stack([idx[0] * self.n_out + idx[1], idx[2]], dim=0)

        self.register_buffer("psi_idx", idx.contiguous(), persistent=False)
        self.register_buffer("psi_vals", vals.contiguous(), persistent=False)

    def get_psi(self):
        psi = torch.sparse_coo_tensor(self.psi_idx, self.psi_vals, size=(self.kernel_size * self.n_out, self.n_in))
        return psi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-multiply x with the quadrature weights
        x = self.quad_weights * x

        psi = self.get_psi()

        # extract shape
        B, C, _ = x.shape

        # bring into the right shape for the bmm and perform it
        x = x.reshape(B * C, self.n_in).permute(1, 0).contiguous()
        x = torch.mm(psi, x)
        x = x.permute(1, 0).reshape(B, C, self.kernel_size, self.n_out)
        x = x.reshape(B, self.groups, self.groupsize, self.kernel_size, self.n_out)

        # do weight multiplication
        out = torch.einsum("bgckx,gock->bgox", x, self.weight.reshape(self.groups, -1, self.weight.shape[1], self.weight.shape[2]))
        out = out.reshape(out.shape[0], -1, out.shape[-1])

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1)

        return out


class DiscreteContinuousConvTranspose2d(DiscreteContinuousConv):
    """
    Transpose Discrete-continuous convolutions (DISCO) [1] on arbitrary 2d grids as implemented for [2]

    [1] Ocampo J., Price M.A. , McEwen J.D.; Scalable and equivariant spherical CNNs by discrete-continuous (DISCO) convolutions, ICLR (2023), arXiv:2209.13603
    [2] Liu-Schiaffini M., Berner J., Bonev B., Kurth T., Azizzadenesheli K., Anandkumar A.; Neural Operators with Localized Integral and Differential Kernels;  arxiv:2402.16845
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_in: torch.Tensor,
        grid_out: torch.Tensor,
        kernel_shape: Union[int, List[int]],
        n_in: Optional[Tuple[int]] = None,
        n_out: Optional[Tuple[int]] = None,
        quad_weights: Optional[torch.Tensor] = None,
        periodic: Optional[bool] = False,
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        radius_cutoff: Optional[float] = None,
    ):
        super().__init__(in_channels, out_channels, kernel_shape, groups, bias)

        # the instantiator supports convenience constructors for the input and output grids
        if isinstance(grid_in, torch.Tensor):
            assert isinstance(quad_weights, torch.Tensor)
            assert not periodic
        elif isinstance(grid_in, str):
            assert n_in is not None
            assert len(n_in) == 2
            x, wx = _precompute_grid(n_in[0], grid=grid_in, periodic=periodic)
            y, wy = _precompute_grid(n_in[1], grid=grid_in, periodic=periodic)
            x, y = torch.meshgrid(torch.from_numpy(x), torch.from_numpy(y))
            wx, wy = torch.meshgrid(torch.from_numpy(wx), torch.from_numpy(wy))
            grid_in = torch.stack([x.reshape(-1), y.reshape(-1)])
            quad_weights = (wx * wy).reshape(-1)
        else:
            raise ValueError(f"Unknown grid input type of type {type(grid_in)}")

        if isinstance(grid_out, torch.Tensor):
            pass
        elif isinstance(grid_out, str):
            assert n_out is not None
            assert len(n_out) == 2
            x, wx = _precompute_grid(n_out[0], grid=grid_out, periodic=periodic)
            y, wy = _precompute_grid(n_out[1], grid=grid_out, periodic=periodic)
            x, y = torch.meshgrid(torch.from_numpy(x), torch.from_numpy(y))
            grid_out = torch.stack([x.reshape(-1), y.reshape(-1)])
        else:
            raise ValueError(f"Unknown grid output type of type {type(grid_out)}")

        # check that input arrays are valid point clouds in 2D
        assert len(grid_in.shape) == 2
        assert len(grid_out.shape) == 2
        assert len(quad_weights.shape) == 1
        assert grid_in.shape[0] == 2
        assert grid_out.shape[0] == 2

        self.n_in = grid_in.shape[-1]
        self.n_out = grid_out.shape[-1]

        # compute the cutoff radius based on the bandlimit of the input field
        # TODO: Attention - this heuristic is ad-hoc! Make sure to set it yourself!
        if radius_cutoff is None:
            radius_cutoff = 2 / float(math.sqrt(self.n_in) - 1)

        if radius_cutoff <= 0.0:
            raise ValueError("Error, radius_cutoff has to be positive.")

        # integration weights
        self.register_buffer("quad_weights", quad_weights, persistent=False)

        # precompute the transposed tensor
        idx, vals = _precompute_convolution_tensor_2d(
            grid_out, grid_in, self.kernel_shape, quad_weights, radius_cutoff=radius_cutoff, periodic=periodic, transpose_normalization=True
        )

        # to improve performance, we make psi a matrix by merging the first two dimensions
        # This has to be accounted for in the forward pass
        idx = torch.stack([idx[0] * self.n_out + idx[2], idx[1]], dim=0)

        self.register_buffer("psi_idx", idx.contiguous(), persistent=False)
        self.register_buffer("psi_vals", vals.contiguous(), persistent=False)

    def get_psi(self):
        psi = torch.sparse_coo_tensor(self.psi_idx, self.psi_vals, size=(self.kernel_size * self.n_out, self.n_in))
        return psi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-multiply x with the quadrature weights
        x = self.quad_weights * x

        psi = self.get_psi()

        # extract shape
        B, C, _ = x.shape

        # bring into the right shape for the bmm and perform it
        x = x.reshape(B * C, self.n_in).permute(1, 0).contiguous()
        x = torch.mm(psi, x)
        x = x.permute(1, 0).reshape(B, C, self.kernel_size, self.n_out)
        x = x.reshape(B, self.groups, self.groupsize, self.kernel_size, self.n_out)

        # do weight multiplication
        out = torch.einsum("bgckx,gock->bgox", x, self.weight.reshape(self.groups, -1, self.weight.shape[1], self.weight.shape[2]))
        out = out.reshape(out.shape[0], -1, out.shape[-1])

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1)

        return out


class EquidistantDiscreteContinuousConv2d(DiscreteContinuousConv):
    """
    Transpose Discrete-continuous convolutions (DISCO) [1] on equidistant 2d grids as implemented for [2]. This implementation maps to 2d convolution kernels which makes it more efficient than the unstructured implementation above.

    [1] Ocampo J., Price M.A. , McEwen J.D.; Scalable and equivariant spherical CNNs by discrete-continuous (DISCO) convolutions, ICLR (2023), arXiv:2209.13603
    [2] Liu-Schiaffini M., Berner J., Bonev B., Kurth T., Azizzadenesheli K., Anandkumar A.; Neural Operators with Localized Integral and Differential Kernels;  arxiv:2402.16845
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        kernel_shape: Union[int, List[int]],
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        radius_cutoff: Optional[float] = None,
        periodic: Optional[bool] = False,
        padding_mode: str = "circular",
        **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_shape, groups, bias)

        self.padding_mode = padding_mode

        # compute the cutoff radius based on the assumption that the grid is [-1, 1]^2
        # this still assumes a quadratic domain
        if radius_cutoff is None:
            radius_cutoff = 2 / float(max(*in_shape))

        if radius_cutoff <= 0.0:
            raise ValueError("Error, radius_cutoff has to be positive.")

        # compute how big the discrete kernel needs to be for the 2d convolution kernel to work
        self.psi_local_size = math.floor(2*radius_cutoff * max(*in_shape) / 2) + 1

        # psi_local is essentially the support of the hat functions evaluated locally
        x = torch.linspace(-radius_cutoff, radius_cutoff, self.psi_local_size)
        x, y = torch.meshgrid(x, x)
        grid_in = torch.stack([x.reshape(-1), y.reshape(-1)])
        quad_weights = torch.ones(self.psi_local_size * self.psi_local_size) / float(in_shape[0] * in_shape[1])
        grid_out = torch.Tensor([[0.0], [0.0]])

        # precompute psi using conventional routines onto the local grid
        idx, vals = _precompute_convolution_tensor_2d(grid_in, grid_out, self.kernel_shape, quad_weights, radius_cutoff=radius_cutoff, periodic=False)

        # extract the local psi
        psi_loc = torch.zeros(self.kernel_size, self.psi_local_size*self.psi_local_size)
        for ie in range(len(vals)):
            f = idx[0, ie]; j = idx[2, ie]; v = vals[ie]
            psi_loc[f, j] = v

        # compute local version of the filter matrix
        psi_loc = psi_loc.reshape(self.kernel_size, self.psi_local_size, self.psi_local_size)
        # # normalization by the quadrature weights
        # psi_loc = 4.0 * psi_loc / float(in_shape[0]*in_shape[1])

        self.register_buffer("psi_loc", psi_loc, persistent=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        kernel = torch.einsum("kxy,ogk->ogxy", self.psi_loc, self.weight)

        left_pad = self.psi_local_size // 2
        right_pad = (self.psi_local_size+1) // 2 - 1
        x = nn.functional.pad(x, (left_pad, right_pad, left_pad, right_pad), mode=self.padding_mode)
        out = nn.functional.conv2d(x, kernel, self.bias, stride=1, dilation=1, padding=0, groups=self.groups)

        return out