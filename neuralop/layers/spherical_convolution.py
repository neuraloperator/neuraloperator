from typing import List, Optional, Union

import torch
from torch import nn
from torch_harmonics import RealSHT, InverseRealSHT

import tensorly as tl
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor

from neuralop.utils import validate_scaling_factor
from .base_spectral_conv import BaseSpectralConv
from .spectral_convolution import SubConv

tl.set_backend("pytorch")
use_opt_einsum("optimal")

einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _contract_dense(x, weight, separable=False, dhconv=True):
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    if dhconv:
        weight_syms.pop()

    eq = "".join(x_syms) + "," + "".join(weight_syms) + "->" + "".join(out_syms)

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    return tl.einsum(eq, x, weight)


def _contract_dense_separable(x, weight, separable=True, dhconv=False):
    if not separable:
        raise ValueError("This function is only for separable=True")
    if dhconv:
        return x * weight.unsqueeze(-1)
    return x * weight


def _contract_cp(x, cp_weight, separable=False, dhconv=True):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order + 1]
    out_syms = list(x_syms)

    if separable:
        factor_syms = [einsum_symbols[1] + rank_sym]  # in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1] + rank_sym, out_sym + rank_sym]  # in, out

    if dhconv:
        factor_syms += [xs + rank_sym for xs in x_syms[2:-1]]  # x, y, ...
    else:
        factor_syms += [xs + rank_sym for xs in x_syms[2:]]  # x, y, ...

    eq = (
        x_syms + "," + rank_sym + "," + ",".join(factor_syms) + "->" + "".join(out_syms)
    )

    return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)


def _contract_tucker(x, tucker_weight, separable=False, dhconv=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        factor_syms = [xs + rs for (xs, rs) in zip(x_syms[1:], core_syms)]  # x, y, ...

    elif dhconv:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # out, in
        factor_syms += [
            xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])
        ]  # x, y, ...
    else:
        core_syms = einsum_symbols[order + 1 : 2 * order + 1]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # out, in
        factor_syms += [
            xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])
        ]  # x, y, ...

    eq = (
        x_syms
        + ","
        + core_syms
        + ","
        + ",".join(factor_syms)
        + "->"
        + "".join(out_syms)
    )

    return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def _contract_tt(x, tt_weight, separable=False, dhconv=False):
    order = tl.ndim(x)

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])  # no batch-size
    if not separable:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)

    if dhconv:
        weight_syms = weight_syms[:-1]  # no batch-size, no y dim

    rank_syms = list(einsum_symbols[order + 1 :])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i + 1]])
    eq = (
        "".join(x_syms)
        + ","
        + ",".join("".join(f) for f in tt_syms)
        + "->"
        + "".join(out_syms)
    )

    return tl.einsum(eq, x, *tt_weight.factors)


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input
        (factorized)
    separable : bool
        whether to use the separable implementation of contraction. This arg is
        only checked when `implementation=reconstructed`.

    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == "reconstructed":
        if separable:
            print("SEPARABLE")
            return _contract_dense_separable
        else:
            return _contract_dense
    elif implementation == "factorized":
        if torch.is_tensor(weight):
            return _contract_dense
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower().endswith("dense"):
                return _contract_dense
            elif weight.name.lower().endswith("tucker"):
                return _contract_tucker
            elif weight.name.lower().endswith("tt"):
                return _contract_tt
            elif weight.name.lower().endswith("cp"):
                return _contract_cp
            else:
                raise ValueError(f"Got unexpected factorized weight type {weight.name}")
        else:
            raise ValueError(
                f"Got unexpected weight type of class {weight.__class__.__name__}"
            )
    else:
        raise ValueError(
            f'Got implementation={implementation}, expected "reconstructed" or '
            f'"factorized"'
        )


Number = Union[int, float]


class SHT(nn.Module):
    """A wrapper for the Spherical Harmonics transform 

    Allows to call it with an interface similar to that of FFT
    """
    def __init__(self, dtype=torch.float32, device=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self._SHT_cache = nn.ModuleDict()
        self._iSHT_cache = nn.ModuleDict()

    def sht(self, x, s=None, norm="ortho", grid="equiangular"):
        *_, height, width = x.shape # height = latitude, width = longitude
        if s is None:
            if grid == "equiangular":
                modes_width = height // 2
            else:
                modes_width = height
            modes_height = height
        else:
            modes_height, modes_width = s

        cache_key = f"{height}_{width}_{modes_height}_{modes_width}_{norm}_{grid}"

        try:
            sht = self._SHT_cache[cache_key]
        except KeyError:
            sht = (
                RealSHT(
                    nlat=height,
                    nlon=width,
                    lmax=modes_height,
                    mmax=modes_width,
                    grid=grid,
                    norm=norm
                )
                .to(device=x.device)
                .to(dtype=self.dtype)
            )
            self._SHT_cache[cache_key] = sht
        
        return sht(x)


    def isht(self, x, s=None, norm="ortho", grid="equiangular"):
        *_, modes_height, modes_width = x.shape # height = latitude, width = longitude
        if s is None:
            if grid == "equiangular":
                width = modes_width*2
            else:
                width = modes_width
            height = modes_height
        else:
            height, width = s

        cache_key = f"{height}_{width}_{modes_height}_{modes_width}_{norm}_{grid}"

        try:
            isht = self._iSHT_cache[cache_key]
        except KeyError:
            isht = (
                InverseRealSHT(
                    nlat=height,
                    nlon=width,
                    lmax=modes_height,
                    mmax=modes_width,
                    grid=grid,
                    norm=norm
                )
                .to(device=x.device)
                .to(dtype=self.dtype)
            )
            self._iSHT_cache[cache_key] = isht
        
        return isht(x)


class SphericalConv(BaseSpectralConv):
    """Spherical Convolution, base class for the SFNO [1]_
    
    Parameters
    ----------
    sht_norm : str, {'ortho'}
    sht_grids : str or str list, default is "equiangular", {"equiangular", "legendre-gauss"}
                * If str, the same grid is used for all layers
                * If list, should have n_layers + 1 values, corresponding to the input and output grid of each layer
                  e.g. for 1 layer, ["input_grid", "output_grid"]

    See SpectralConv for full list of other parameters

    [1] : Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere,
          Boris Bonev, Thorsten Kurth, Christian Hundt, Jaideep Pathak, Maximilian Baust, Karthik Kashinath, Anima Anandkumar,
          ICML 2023.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        incremental_n_modes=None,
        bias=True,
        n_layers=1,
        separable=False,
        output_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        # fno_block_precision="full",
        rank=0.5,
        factorization="cp",
        implementation="reconstructed",
        fixed_rank_modes=False,
        joint_factorization=False,
        decomposition_kwargs=dict(),
        init_std="auto",
        sht_norm="ortho",
        sht_grids="equiangular",
        device=None,
        dtype=torch.float32,
    ):
        super().__init__(dtype=dtype, device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.joint_factorization = joint_factorization

        # We index quadrants only
        # n_modes is the total number of modes kept along each dimension
        # half_n_modes is half of that except in the last mode, corresponding to
        # the number of modes to keep in *each* quadrant for each dim
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        self.order = len(n_modes)

        half_total_n_modes = list(n_modes)
        half_total_n_modes[-1] = half_total_n_modes[-1] // 2
        self.half_total_n_modes = half_total_n_modes

        # WE use half_total_n_modes to build the full weights
        # During training we can adjust incremental_n_modes which will also
        # update half_n_modes
        # So that we can train on a smaller part of the Fourier modes
        # and total weights
        self.incremental_n_modes = incremental_n_modes

        self.rank = rank
        self.factorization = factorization
        self.n_layers = n_layers
        self.implementation = implementation

        self.output_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(output_scaling_factor, self.order, n_layers)

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels))**0.5
        else:
            init_std = init_std

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None

        # Make sure we are using a Complex Factorized Tensor to parametrize the conv
        if factorization is None:
            factorization = "Dense"  # No factorization
        if not factorization.lower().startswith("complex"):
            factorization = f"Complex{factorization}"

        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal "
                    f"to out_channels, but got in_channels={in_channels} "
                    f"and out_channels={out_channels}",
                )
            weight_shape = (in_channels, *half_total_n_modes[:-1])
        else:
            weight_shape = (in_channels, out_channels, *half_total_n_modes[:-1])
        self.separable = separable

        if joint_factorization:
            self.weight = FactorizedTensor.new(
                (self.n_layers, *weight_shape),
                rank=self.rank,
                factorization=factorization,
                fixed_rank_modes=fixed_rank_modes,
                **decomposition_kwargs,
            )
            self.weight.normal_(0, init_std)
        else:
            self.weight = nn.ModuleList(
                [
                    FactorizedTensor.new(
                        weight_shape,
                        rank=self.rank,
                        factorization=factorization,
                        fixed_rank_modes=fixed_rank_modes,
                        **decomposition_kwargs,
                    )
                    for _ in range(n_layers)
                ]
            )
            for w in self.weight:
                w.normal_(0, init_std)
        self._contract = get_contract_fun(
            self.weight[0], implementation=implementation, separable=separable
        )

        if bias:
            self.bias = nn.Parameter(
                init_std
                * torch.randn(*((n_layers, self.out_channels) + (1,) * self.order))
            )
        else:
            self.bias = None

        self.sht_norm = sht_norm
        if isinstance(sht_grids, str):
            sht_grids = [sht_grids]*(self.n_layers + 1)
        self.sht_grids = sht_grids
        self.sht_handle = SHT(dtype=self.dtype, device=self.device)

    def _get_weight(self, index):
        if self.incremental_n_modes is not None:
            return self.weight[index][self.weight_slices]
        else:
            return self.weight[index]

    @property
    def incremental_n_modes(self):
        return self._incremental_n_modes

    @incremental_n_modes.setter
    def incremental_n_modes(self, incremental_n_modes):
        if incremental_n_modes is None:
            self._incremental_n_modes = None
            self.half_n_modes = [m // 2 for m in self.n_modes]

        else:
            if isinstance(incremental_n_modes, int):
                self._incremental_n_modes = [incremental_n_modes] * len(self.n_modes)
            else:
                if len(incremental_n_modes) == len(self.n_modes):
                    self._incremental_n_modes = incremental_n_modes
                else:
                    raise ValueError(
                        f"Provided {incremental_n_modes} for actual"
                        f" n_modes={self.n_modes}."
                    )
            self.weight_slices = [slice(None)] * 2 + [
                slice(None, n // 2) for n in self._incremental_n_modes
            ]
            self.half_n_modes = [m // 2 for m in self._incremental_n_modes]
    
    def transform(self, x, layer_index=0, output_shape=None):
        *_, in_height, in_width = x.shape

        if self.output_scaling_factor is not None and output_shape is None:
            height = round(in_height * self.output_scaling_factor[layer_index][0])
            width = round(in_width * self.output_scaling_factor[layer_index][1])
        elif output_shape is not None:
            height, width = output_shape[0], output_shape[1]
        else:
            height, width = in_height, in_width

        # Return the identity if the resolution and grid of the input and output are the same
        if ((in_height, in_width) == (height, width)) and (self.sht_grids[layer_index] == self.sht_grids[layer_index+1]):
            return x
        else:
            coefs = self.sht_handle.sht(x, s=self.n_modes, norm=self.sht_norm, grid=self.sht_grids[layer_index])
            return self.sht_handle.isht(coefs, s=(height, width), norm=self.sht_norm, grid=self.sht_grids[layer_index + 1])

    def forward(self, x, indices=0, output_shape=None):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
        indices : int, default is 0
            if joint_factorization, index of the layers for n_layers > 1

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, height, width = x.shape

        if self.output_scaling_factor is not None and output_shape is None:
            scaling_factors = self.output_scaling_factor[indices]
            height = round(height * scaling_factors[0])
            width = round(width * scaling_factors[1])
        elif output_shape is not None:
            height, width = output_shape[0], output_shape[1]

        out_fft = self.sht_handle.sht(x, s=self.half_total_n_modes, norm=self.sht_norm,
                                      grid=self.sht_grids[indices])

        # xp = torch.zeros_like(x)
        # xp[..., : self.modes_lat_local, : self.modes_lon_local] = self._contract(
        #     x[..., : self.modes_lat_local, : self.modes_lon_local],
        #     self.weight,
        #     separable=self.separable,
        #     operator_type=self.operator_type,
        # )

        # upper block (truncate high freq)
        out_fft = self._contract(
            out_fft[:, :, :self.half_total_n_modes[0], :self.half_total_n_modes[1]],
            self._get_weight(indices),
            separable=self.separable,
            dhconv=True,
        )

        x = self.sht_handle.isht(out_fft, s=(height, width), norm=self.sht_norm,
                                 grid=self.sht_grids[indices+1])

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution

        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single convolution is parametrized, directly use the main class."
            )

        return SubConv(self, indices)

    def __getitem__(self, indices):
        return self.get_conv(indices)
