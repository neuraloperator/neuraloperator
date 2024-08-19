import math
import numpy
import torch

from typing import Union, List, Optional, Tuple
from functools import partial

# import the base class from torch-harmonics
from torch_harmonics.quadrature import _precompute_grid
from torch_harmonics.convolution import _compute_support_vals_isotropic, _compute_support_vals_anisotropic
from torch_harmonics.convolution import DiscreteContinuousConv

def _precompute_convolution_tensor_2d(grid_in, grid_out, kernel_shape, radius_cutoff=0.01, periodic=False):
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
        periodic_diffs = torch.where(diffs > 0.0, diffs-1, diffs+1)
        diffs = torch.where(diffs.abs() < periodic_diffs.abs(), diffs, periodic_diffs)


    r = torch.sqrt(diffs[0] ** 2 + diffs[1] ** 2)
    phi = torch.arctan2(diffs[1], diffs[0]) + torch.pi

    idx, vals = kernel_handle(r, phi)
    idx = idx.permute(1, 0)

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
        # TODO: this heuristic is ad-hoc! Verify that we do the right one
        if radius_cutoff is None:
            radius_cutoff = 2 * (self.kernel_shape[0] + 1) / float(math.sqrt(self.n_in) - 1)

        if radius_cutoff <= 0.0:
            raise ValueError("Error, radius_cutoff has to be positive.")

        # integration weights
        self.register_buffer("quad_weights", quad_weights, persistent=False)

        idx, vals = _precompute_convolution_tensor_2d(grid_in, grid_out, self.kernel_shape, radius_cutoff=radius_cutoff, periodic=periodic)

        # to improve performance, we make psi a matrix by merging the first two dimensions
        # This has to be accounted for in the forward pass
        idx = torch.stack([idx[0]*self.n_out + idx[1], idx[2]], dim=0)

        self.register_buffer("psi_idx", idx.contiguous(), persistent=False)
        self.register_buffer("psi_vals", vals.contiguous(), persistent=False)

    def get_psi(self):
        psi = torch.sparse_coo_tensor(self.psi_idx, self.psi_vals, size=(self.kernel_size*self.n_out, self.n_in))
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
        # TODO: this heuristic is ad-hoc! Verify that we do the right one
        if radius_cutoff is None:
            radius_cutoff = 2 * (self.kernel_shape[0] + 1) / float(math.sqrt(self.n_in) - 1)

        if radius_cutoff <= 0.0:
            raise ValueError("Error, radius_cutoff has to be positive.")

        # integration weights
        self.register_buffer("quad_weights", quad_weights, persistent=False)

        # precompute the transposed tensor
        idx, vals = _precompute_convolution_tensor_2d(grid_out, grid_in, self.kernel_shape, radius_cutoff=radius_cutoff, periodic=periodic)

        # to improve performance, we make psi a matrix by merging the first two dimensions
        # This has to be accounted for in the forward pass
        idx = torch.stack([idx[0]*self.n_out + idx[2], idx[1]], dim=0)

        self.register_buffer("psi_idx", idx.contiguous(), persistent=False)
        self.register_buffer("psi_vals", vals.contiguous(), persistent=False)

    def get_psi(self):
        psi = torch.sparse_coo_tensor(self.psi_idx, self.psi_vals, size=(self.kernel_size*self.n_out, self.n_in))
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