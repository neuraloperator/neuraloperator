"""Klein-Gordon Spectral Convolution Layer

A physics-constrained spectral convolution layer that encodes the
Klein-Gordon dispersion relation as a learnable spectral filter.

Instead of learning arbitrary complex weights in Fourier space
(as in the standard FNO SpectralConv), this layer uses the exact
solution operator of the Klein-Gordon equation:

    H(k) = cos(T * sqrt(c^2 |k|^2 + chi^2))

with only 3 learnable scalar parameters (T, c, chi), plus a linear
channel mixing matrix. This makes it dramatically more
parameter-efficient for wave-type (hyperbolic) PDEs.

Mathematical background
-----------------------
The Klein-Gordon equation d^2u/dt^2 = c^2 nabla^2 u - chi^2 u has
the dispersion relation omega^2 = c^2 |k|^2 + chi^2. Its Green's
function in d dimensions is the Matern kernel family [1]_:

- Matern(nu=0.5) = exp(-r/l)            ... 1D KG Green's function
- Matern(nu=1.5) = (1+sqrt(3)r/l)e^(-sqrt(3)r/l)  ... 3D KG
- Matern(nu -> inf) = exp(-r^2/2l^2)    ... RBF (diffusion limit)

The standard RBF kernel used in SVMs and GPs is the diffusion
(infinite smoothness) limit. The KG filter offers finite-smoothness
alternatives motivated by wave physics.

References
----------
.. [1] Whittle, P. "On stationary processes in the plane" (1954).
    Biometrika, 41(3-4), 434-449.

.. [2] Rasmussen, C.E. & Williams, C.K.I. "Gaussian Processes for
    Machine Learning" (2006). MIT Press. Chapter 4.

.. [3] Li, Z. et al. "Fourier Neural Operator for Parametric Partial
    Differential Equations" (2021). ICLR 2021.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from .base_spectral_conv import BaseSpectralConv

Number = Union[int, float]


class KGSpectralConv(BaseSpectralConv):
    """Klein-Gordon Spectral Convolution

    Applies a physics-constrained spectral filter based on the Klein-Gordon
    dispersion relation: ``H(k) = cos(T * sqrt(c^2 |k|^2 + chi^2))``.

    This layer is a drop-in alternative to
    :class:`~neuralop.layers.spectral_convolution.SpectralConv` for
    wave-type PDEs. It trades the FNO's unconstrained Fourier weights
    (O(C_in * C_out * prod(n_modes)) parameters) for a physics-constrained
    filter with only 3 scalar parameters plus a channel mixing matrix
    (O(C_in * C_out + 3) parameters).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    n_modes : int or tuple of int
        Number of Fourier modes to retain along each spatial dimension.
        For real-valued data, the last dimension is automatically adjusted
        for the real FFT redundancy.
    init_T : float, optional
        Initial propagation time parameter. Default: 1.0.
    init_c : float, optional
        Initial wave speed parameter. Default: 1.0.
    init_chi : float, optional
        Initial mass/dispersion parameter. Default: 0.1.
    per_channel : bool, optional
        If True, learn separate (T, c, chi) per output channel.
        Default: False (shared across channels).
    bias : bool, optional
        If True, add a learnable bias term. Default: True.
    complex_data : bool, optional
        If True, input data is complex-valued in the spatial domain.
        Default: False.
    fft_norm : str, optional
        Normalization mode for FFT. Default: ``'forward'``.
    device : torch.device or None, optional
        Device for parameters. Default: None.

    Notes
    -----
    **When to use this layer:**

    - The underlying PDE is hyperbolic (wave equation, Klein-Gordon,
      Maxwell's equations, linearized Euler)
    - Training data is limited and you need parameter efficiency
    - You want a physically interpretable spectral filter

    **When to prefer standard SpectralConv:**

    - The PDE is parabolic (diffusion, heat equation)
    - You have abundant training data
    - Maximum expressiveness is more important than parameter efficiency

    **Dispersion relation limits:**

    - chi = 0: reduces to the wave equation filter cos(T c |k|)
    - T = 0: identity operator (no time evolution)
    - c -> 0: uniform damping cos(T chi) independent of k

    Examples
    --------
    >>> layer = KGSpectralConv(in_channels=3, out_channels=3, n_modes=(16,))
    >>> x = torch.randn(4, 3, 64)   # batch=4, channels=3, nx=64
    >>> y = layer(x)
    >>> y.shape
    torch.Size([4, 3, 64])

    >>> # 2D case
    >>> layer_2d = KGSpectralConv(in_channels=1, out_channels=1, n_modes=(16, 16))
    >>> x2d = torch.randn(2, 1, 64, 64)
    >>> y2d = layer_2d(x2d)
    >>> y2d.shape
    torch.Size([2, 1, 64, 64])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Union[int, Tuple[int, ...]],
        init_T: float = 1.0,
        init_c: float = 1.0,
        init_chi: float = 0.1,
        per_channel: bool = False,
        bias: bool = True,
        complex_data: bool = False,
        fft_norm: str = "forward",
        device=None,
    ):
        super().__init__(device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.complex_data = complex_data
        self.fft_norm = fft_norm

        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = list(n_modes)
        self.order = len(self._n_modes)

        # KG dispersion parameters (learnable, stored in log-space for positivity)
        param_shape = (out_channels,) if per_channel else (1,)
        self.log_T = nn.Parameter(
            torch.full(param_shape, float(np.log(max(init_T, 1e-8))), device=device)
        )
        self.log_c = nn.Parameter(
            torch.full(param_shape, float(np.log(max(init_c, 1e-8))), device=device)
        )
        self.log_chi = nn.Parameter(
            torch.full(param_shape, float(np.log(max(init_chi, 1e-8))), device=device)
        )

        # Channel mixing weights
        init_std = (2 / (in_channels + out_channels)) ** 0.5
        self.channel_weight = nn.Parameter(
            init_std * torch.randn(in_channels, out_channels, device=device)
        )

        if bias:
            self.bias = nn.Parameter(
                init_std
                * torch.randn(
                    *((out_channels,) + (1,) * self.order), device=device
                )
            )
        else:
            self.bias = None

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = list(n_modes)

    def _compute_kg_filter(self, spatial_sizes):
        """Compute the KG spectral filter H(k) on the FFT frequency grid.

        Parameters
        ----------
        spatial_sizes : list of int
            Spatial dimensions of the input (e.g. [nx] or [nx, ny]).

        Returns
        -------
        H : torch.Tensor
            Real-valued spectral filter of shape
            ``(out_channels or 1, k1, k2, ...)``.
        """
        T = torch.exp(self.log_T)
        c = torch.exp(self.log_c)
        chi = torch.exp(self.log_chi)

        # Build wavenumber grid
        freq_components = []
        for i, size in enumerate(spatial_sizes):
            if i == len(spatial_sizes) - 1 and not self.complex_data:
                freqs = torch.fft.rfftfreq(size, device=T.device) * 2 * np.pi
            else:
                freqs = torch.fft.fftfreq(size, device=T.device) * 2 * np.pi
            freq_components.append(freqs)

        grids = torch.meshgrid(*freq_components, indexing="ij")
        k_squared = sum(g**2 for g in grids)  # |k|^2

        # Reshape parameters for broadcasting: (channels, 1, 1, ...)
        ndim = len(spatial_sizes)
        shape = (-1,) + (1,) * ndim
        c_sq = c.view(shape) ** 2
        chi_sq = chi.view(shape) ** 2
        T_val = T.view(shape)

        # omega(k) = sqrt(c^2 |k|^2 + chi^2)
        omega = torch.sqrt(c_sq * k_squared.unsqueeze(0) + chi_sq)

        # H(k) = cos(T * omega(k))
        H = torch.cos(T_val * omega)

        return H

    @property
    def n_kernel_params(self):
        """Number of learnable parameters in the spectral filter (T, c, chi)."""
        return self.log_T.numel() + self.log_c.numel() + self.log_chi.numel()

    @property
    def n_total_params(self):
        """Total learnable parameters including channel mixing and bias."""
        total = self.n_kernel_params + self.channel_weight.numel()
        if self.bias is not None:
            total += self.bias.numel()
        return total

    def forward(
        self,
        x: torch.Tensor,
        output_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Apply the Klein-Gordon spectral filter.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, in_channels, n1, ..., nd)``.
        output_shape : tuple of int or None, optional
            If provided, the output spatial dimensions will be resized
            to this shape via the inverse FFT. Default: None (same size
            as input).

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, out_channels, n1, ..., nd)``
            (or ``output_shape`` if provided).
        """
        batchsize, channels, *mode_sizes = x.shape
        fft_dims = list(range(-self.order, 0))

        # Forward FFT
        if self.complex_data:
            x_hat = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
        else:
            x_hat = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)

        # Channel mixing: (batch, in_ch, k...) -> (batch, out_ch, k...)
        # Cast weight to match x_hat dtype (real -> complex is fine)
        w = self.channel_weight.to(x_hat.dtype)
        out_hat = torch.einsum("bi...,io->bo...", x_hat, w)

        # Apply KG spectral filter (real-valued, broadcasts over batch)
        H = self._compute_kg_filter(mode_sizes)
        out_hat = out_hat * H.unsqueeze(0).to(out_hat.dtype)

        # Inverse FFT
        out_sizes = output_shape if output_shape is not None else mode_sizes
        if self.complex_data:
            x = torch.fft.ifftn(
                out_hat, s=out_sizes, dim=fft_dims, norm=self.fft_norm
            )
        else:
            x = torch.fft.irfftn(
                out_hat, s=out_sizes, dim=fft_dims, norm=self.fft_norm
            )

        if self.bias is not None:
            x = x + self.bias

        return x

    def extra_repr(self) -> str:
        T = torch.exp(self.log_T).detach()
        c = torch.exp(self.log_c).detach()
        chi = torch.exp(self.log_chi).detach()
        T_str = f"{T.item():.4g}" if T.numel() == 1 else str(T.tolist())
        c_str = f"{c.item():.4g}" if c.numel() == 1 else str(c.tolist())
        chi_str = f"{chi.item():.4g}" if chi.numel() == 1 else str(chi.tolist())
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"n_modes={self._n_modes}, "
            f"T={T_str}, c={c_str}, chi={chi_str}, "
            f"kernel_params={self.n_kernel_params}, "
            f"total_params={self.n_total_params}"
        )
