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
        
    # ------------------------------- helpers -------------------------------
        
        
    # ------------------------------- forward -------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
         pass