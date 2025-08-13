from __features__ import Feature
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

try: 
    from pytorch_wavelets import DWT1D, IDWT1D
    from pytorch_waqvelets import DWT, IDWT
    _HAS_PTW = True
except Exception: 
    _HAS_PTW = False
    
try:
    import pywt
    from ptwt.conv_transform_3 import wavedec3 as ptwt_wavedec3
    from ptwt.conv_transform_3 import waverec3 as ptwt_waverec3
    _HAS_PYWT_PTWT = True
except Exception:
    _HAS_PYWT_PTWT = False
    
def _ensure_tuple_size(size: Union[int, Sequence[int]], n_dim: int) -> Tuple[int, ...]:
    if n_dim == 1:
        if isinstance(size, int):
            return (size,)
        raise ValueError("For n_dim=1, size must be an int (signal length).")
    else:
        if not (isinstance(size, (list, tuple)) and len(size) == n_dim):
            raise ValueError(f"For n_dim={n_dim}, size must be a list/tuple of length {n_dim}.")
        return tuple(int(s) for s in size)


class SpectralConvWavelet(nn.Module):
    """Implements Unified Wavelet-based Spectral Convolution for n_dim ∈ {1,2,3}.
    described in [1]_

    Parameters
    ----------
    in_channels, out_channels : int
    level : int
        Decomposition levels (we act on the last level only).
    size : int | Tuple[int, ...]
        Nominal spatial size. For n_dim=1 provide an int; for 2D/3D a tuple/list.
    n_dim : {1,2,3}
        Dimensionality selector.
    wavelet : str
        Wavelet name (e.g., 'db4'). For 3D we pass pywt.Wavelet(wavelet) to ptwt.
    mode : str
        Boundary mode (e.g., 'symmetric' for 1D/2D; 'periodization' or 'symmetric' for 3D).

    .. [1] : Tripura, T., and Chakraborty, S. “Wavelet neural operator: a neural operator for parametric partial differential equations” (2022). arXiv preprint arXiv:2205.02191, https://arxiv.org/pdf/2205.02191.

    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        level: int,
        size: Union[int, Sequence[int]],
        n_dim: int,
        wavelet: str = "db4",
        mode: str = "symmetric",
    ) -> None:
        super().__init__()
        
        if n_dim not in (1, 2, 3):
            raise ValueError("n_dim must be 1, 2, or 3")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels/out_channels must be positive")
        if level < 1:
            raise ValueError("level must be >= 1")
        
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.level = int(level)
        self.n_dim = int(n_dim)
        self.wavelet = wavelet
        self.mode = mode
        self.size = _ensure_tuple_size(size, self.n_dim)
        
        # Dependency checks per-dimension
        if self.n_dim in (1, 2) and not _HAS_PTW:
            raise ImportError("pytorch_wavelets is required for n_dim=1 or 2")
        if self.n_dim == 3 and not _HAS_PYWT_PTWT:
            raise ImportError("pywt + ptwt are required for n_dim=3")
        
        # Dependency checks per-dimension
        if self.n_dim in (1, 2) and not _HAS_PTW:
            raise ImportError("pytorch_wavelets is required for n_dim=1 or 2")
        if self.n_dim == 3 and not _HAS_PYWT_PTWT:
            raise ImportError("pywt + ptwt are required for n_dim=3")
        
        # Initialization scale
        self.scale = 1.0 / (self.in_channels * self.out_channels)
        
        # Infer last-level approx subband shape to allocate weights
        with torch.no_grad():
            if self.n_dim == 1:
                dummy = torch.randn(1, 1, self.size[0])
                dwt = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode)
                yl, _ = dwt(dummy)
                self.modes = (yl.shape[-1],)
                # Weights: approx (A) and detail (D)
                self.weights_A = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_D = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))

            elif self.n_dim == 2:
                dummy = torch.randn(1, 1, *self.size)
                dwt = DWT(wave=self.wavelet, J=self.level, mode=self.mode)
                yl, _ = dwt(dummy)
                self.modes = (yl.shape[-2], yl.shape[-1])
                # Weights: approx (A) and H/V/D details
                self.weights_A = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_H = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_V = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_D = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))

            else:  # n_dim == 3
                # ptwt expects tensors; we can probe with a spatial-only dummy, shapes match coefficient dims
                dummy = torch.randn(*self.size).unsqueeze(0)  # (1, Z, Y, X) no channel dim
                mode_data = ptwt_wavedec3(dummy, pywt.Wavelet(self.wavelet), level=self.level, mode=self.mode)
                A = mode_data[0]  # approx
                self.modes = (A.shape[-3], A.shape[-2], A.shape[-1])
                # Weights: approx + 7 detail combinations
                self.weights_A   = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_aad = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_ada = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_add = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_daa = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_dad = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_dda = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                self.weights_ddd = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, *self.modes))
                
        # Precompute einsum pattern for mul
        self._einsum = _einsum_pattern(self.n_dim)

    # ------------------------------- helpers -------------------------------
        
        
    # ------------------------------- forward -------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
         pass