import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Union, Literal
import math

from functools import partial
from scipy.special import eval_legendre
from sympy import Poly, legendre, Symbol, chebyshevt


from ..layers.mwno_block import MWNO_CZ

class MWNO(nn.Module):
    """
    Reference:
    Gupta, G., Xiao, X. and Bogdan, P., 2021.
    Multiwavelet-based operator learning for differential equations.
    Advances in neural information processing systems, 34, pp.24048-24062.

    @article{gupta2021multiwavelet,
      title={Multiwavelet-based operator learning for differential equations},
      author={Gupta, Gaurav and Xiao, Xiongye and Bogdan, Paul},
      journal={Advances in neural information processing systems},
      volume={34},
      pages={24048--24062},
      year={2021}
    }


    Description: Multiwavelet Neural Operator (MWNO) is a neural operator framework
    that employs multiwavelet bases to represent operators in localized, multiscale domains.
    By leveraging the orthogonality, compact support, and multi-resolution properties of multiwavelets,
    MWNO effectively captures both global structures and local irregularities,
    making it particularly well-suited for learning operators from data with high fluctuations.
    This enables end-to-end learning of mappings between infinite-dimensional function spaces and
    supports accurate modeling of complex dynamical systems governed by partial differential equations.



    Parameters
    ----------
    n_modes : Tuple[int] or int
        If int, creates square modes for corresponding dimension (1D: alpha, 2D: (alpha, alpha), 3D: (alpha, alpha, alpha))
        If Tuple, uses specified modes directly
    in_channels : int
        Number of input function channels
    out_channels : int
        Number of output function channels
    k : int, default=3
        Wavelet kernel size
    c : int, default=1
        Number of channels in wavelet transform
    n_layers : int, default=3
        Number of MWNO_CZ layers
    L : int, default=0
        Number of coarsest scale layers to skip
    lifting_channels : int, default=128
        Lifting layer channels, if 0 uses simple linear transformation
    projection_channels : int, default=128
        Projection layer channels, if 0 uses simple linear transformation
    base : str, default='legendre'
        Wavelet basis type ('legendre', 'chebyshev')
    initializer : callable, optional
        Weight initialization function
    
    Examples
    --------
    >>> model_1d = MWNO(alpha=5, in_channels=1, out_channels=1)
    >>> model_2d = MWNO((5, 5), in_channels=1, out_channels=1)
    >>> model_3d = MWNO(alpha=5, n_dim=3, in_channels=1, out_channels=1)
    """
    
    def __init__(
        self,
        n_modes=None,
        alpha=None,
        n_dim=None,
        in_channels: int = 1,
        out_channels: int = 1,
        k: int = 3,
        c: int = 1,
        n_layers: int = 3,
        L: int = 0,
        lifting_channels: int = 0,
        projection_channels: int = 128,
        base: str = 'legendre',
        initializer=None,
        **kwargs
    ):
        super().__init__()
        
        if n_modes is not None:
            if isinstance(n_modes, (tuple, list)):
                self.n_modes = tuple(n_modes)
                self.n_dim = len(self.n_modes)
                alpha = n_modes[0]
            else:
                alpha = n_modes
                self.n_dim = 1
                self.n_modes = (alpha,)
        elif alpha is not None:
            if n_dim is None:
                self.n_dim = 1
            else:
                self.n_dim = n_dim
            self.n_modes = tuple([alpha] * self.n_dim)
        else:
            raise ValueError("Either n_modes or alpha must be specified")
        
        if self.n_dim not in [1, 2, 3]:
            raise ValueError(f"MWNO only supports 1D, 2D, and 3D. Got {self.n_dim}D")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.c = c
        self.n_layers = n_layers
        self.L = L
        self.base = base
        self.alpha = alpha
        
        if self.n_dim == 1:
            ich_multiplier = c * k
        else:
            ich_multiplier = c * k**2
        
        self.lifting = self._build_lifting_layer(in_channels, ich_multiplier, lifting_channels)

        self.mwno_layers = nn.ModuleList([
            MWNO_CZ(k=k, alpha=alpha, L=L, c=c, base=base, n_dim=self.n_dim, initializer=initializer)
            for _ in range(n_layers)
        ])
        
        self.projection = self._build_projection_layer(ich_multiplier, out_channels, projection_channels)
        
        if initializer is not None:
            self._reset_parameters(initializer)
    
    def _build_lifting_layer(self, in_channels, ich_multiplier, lifting_channels):
        """Build lifting layer"""
        if lifting_channels > 0:
            return nn.Sequential(
                nn.Linear(in_channels, lifting_channels),
                nn.ReLU(inplace=True),
                nn.Linear(lifting_channels, ich_multiplier)
            )
        else:
            return nn.Linear(in_channels, ich_multiplier)
    
    def _build_projection_layer(self, ich_multiplier, out_channels, projection_channels):
        """Build projection layer"""
        if projection_channels > 0:
            return nn.Sequential(
                nn.Linear(ich_multiplier, projection_channels),
                nn.ReLU(inplace=True),
                nn.Linear(projection_channels, out_channels)
            )
        else:
            return nn.Sequential(
                nn.Linear(ich_multiplier, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, out_channels)
            )
    
    def _reset_parameters(self, initializer):
        """Reset parameters"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                initializer(module.weight)

    def forward(self, x):
        """
        Forward propagation

        Parameters
        ----------
        x : tensor
            Input tensor with shape:
            - 1D: (B, N, in_channels)
            - 2D: (B, Nx, Ny, in_channels)
            - 3D: (B, Nx, Ny, T, in_channels)

        Returns
        -------
        output : tensor
            Output tensor with shape:
            - 1D: (B, N, out_channels) or (B, N) if out_channels=1
            - 2D: (B, Nx, Ny, out_channels) or (B, Nx, Ny) if out_channels=1
            - 3D: (B, Nx, Ny, T, out_channels) or (B, Nx, Ny, T) if out_channels=1
        """

        # Validate input dimensions
        expected_dims = self.n_dim + 2  # batch + spatial dims + channel
        assert x.ndim == expected_dims, (
            f"Expected {expected_dims}D input for {self.n_dim}D MWNO, "
            f"but got {x.ndim}D tensor with shape {x.shape}"
        )

        # Validate channel dimension
        assert x.shape[-1] == self.in_channels, (
            f"Expected {self.in_channels} input channels, "
            f"but got {x.shape[-1]} channels. Input shape: {x.shape}"
        )

        # Validate spatial dimensions are powers of 2
        if self.n_dim == 1:
            N = x.shape[1]
            assert N & (N - 1) == 0, (
                f"Spatial dimension must be power of 2, but got N={N}. "
                f"Input shape: {x.shape}"
            )
        elif self.n_dim == 2:
            Nx, Ny = x.shape[1], x.shape[2]
            assert Nx & (Nx - 1) == 0 and Ny & (Ny - 1) == 0, (
                f"Spatial dimensions must be powers of 2, but got Nx={Nx}, Ny={Ny}. "
                f"Input shape: {x.shape}"
            )
        elif self.n_dim == 3:
            Nx, Ny = x.shape[1], x.shape[2]
            assert Nx & (Nx - 1) == 0 and Ny & (Ny - 1) == 0, (
                f"Spatial dimensions (Nx, Ny) must be powers of 2, "
                f"but got Nx={Nx}, Ny={Ny}. Input shape: {x.shape}"
            )

        x = self.lifting(x)

        # Reshape to add wavelet dimension with validation
        if self.n_dim == 1:
            B, N, lifted_channels = x.shape
            expected_channels = self.c * self.k
            assert lifted_channels == expected_channels, (
                f"Lifting produced {lifted_channels} channels, "
                f"expected {expected_channels} (c={self.c}, k={self.k})"
            )
            x = x.view(B, N, self.c, self.k)
        elif self.n_dim == 2:
            B, Nx, Ny, lifted_channels = x.shape
            expected_channels = self.c * self.k ** 2
            assert lifted_channels == expected_channels, (
                f"Lifting produced {lifted_channels} channels, "
                f"expected {expected_channels} (c={self.c}, k^2={self.k ** 2})"
            )
            x = x.view(B, Nx, Ny, self.c, self.k ** 2)
        elif self.n_dim == 3:
            B, Nx, Ny, T, lifted_channels = x.shape
            expected_channels = self.c * self.k ** 2
            assert lifted_channels == expected_channels, (
                f"Lifting produced {lifted_channels} channels, "
                f"expected {expected_channels} (c={self.c}, k^2={self.k ** 2})"
            )
            x = x.view(B, Nx, Ny, T, self.c, self.k ** 2)

        # Apply MWNO layers
        for i, layer in enumerate(self.mwno_layers):
            x = layer(x)
            if i < self.n_layers - 1:
                x = F.relu(x)

        # Reshape back and project
        if self.n_dim == 1:
            x = x.view(B, N, -1)
        elif self.n_dim == 2:
            x = x.view(B, Nx, Ny, -1)
        elif self.n_dim == 3:
            x = x.view(B, Nx, Ny, T, -1)

        x = self.projection(x)

        # Validate output channels
        assert x.shape[-1] == self.out_channels, (
            f"Expected {self.out_channels} output channels, "
            f"but got {x.shape[-1]}. Output shape: {x.shape}"
        )

        if self.out_channels == 1:
            return x.squeeze(-1)
        return x


