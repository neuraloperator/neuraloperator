import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Union, Literal, Optional, Callable
import math

from ..layers.mwno_block import MWNO_CZ


class MWNO(nn.Module):
    """N-Dimensional Multiwavelet Neural Operator. The MWNO learns a mapping between
    spaces of functions using multiwavelet bases to represent operators in localized,
    multiscale domains, as described in [1].

    By leveraging orthogonality, compact support, and multi-resolution properties of
    multiwavelets, it effectively captures both global structures and local irregularities.
    This makes it particularly well-suited for learning operators from data with high
    fluctuations and supports accurate modeling of complex dynamical systems governed
    by partial differential equations.

    Main Parameters
    ---------------
    n_modes : Tuple[int, ...] or int
        Fourier modes to retain in each dimension.
        The dimensionality of the MWNO is inferred from len(n_modes).
        If int: Creates (alpha,) for 1D, (alpha, alpha) for 2D, etc.
        Must provide either n_modes or alpha (not both).
    in_channels : int
        Number of input function channels. Determined by the problem.
        For scalar fields: in_channels=1. For vector fields: in_channels=n_components.
    out_channels : int
        Number of output function channels. Determined by the problem.
        Set to 1 for scalar outputs (will auto-squeeze).
    k : int, optional
        Wavelet basis size (number of polynomial basis functions). Default: 3
        k=2: Piecewise linear wavelets, k=3: Cubic wavelets, k=4: Quartic wavelets.
    c : int, optional
        Number of parallel wavelet channels. Default: 1
        Increases model capacity: total wavelet features = c * k^n_dim.
    n_layers : int, optional
        Number of MWNO_CZ transformation layers. Default: 3
        More layers = deeper hierarchical processing.

    Other parameters
    ----------------
    alpha : int, optional
        Alternative way to specify modes (all dimensions use same value).
        Only used if n_modes is not provided.
        Example: alpha=12 with n_dim=2 → modes=(12, 12).
    n_dim : int, optional
        Spatial dimensionality (1, 2, or 3).
        Only needed if using alpha parameter.
        Inferred from n_modes if n_modes is a tuple.
    L : int, optional
        Number of coarsest decomposition levels to skip. Default: 0
        Reduces computation by stopping wavelet decomposition early.
        L=0: Full decomposition to coarsest scale. L=1: Stop 1 level before coarsest.
    lifting_channels : int, optional
        Hidden dimension for lifting layer. Default: 0
        If 0: Direct linear lifting (fast).
        If > 0: Two-layer MLP (in → lifting_channels → wavelet_space).
    projection_channels : int, optional
        Hidden dimension for projection layer. Default: 128
        If 0: Uses default two-layer MLP with hidden_dim=128.
        If > 0: Two-layer MLP (wavelet_space → proj_channels → out).
    base : str, optional
        Polynomial basis for wavelet construction. Default: "legendre"
        Options: "legendre" (uniform weighting, general purpose),
        "chebyshev" (better for boundary-dominated problems).
    initializer : callable, optional
        Custom weight initialization function.
        Applied to all Linear layers if provided.
        Signature: initializer(weight_tensor) -> None. Default: None

    Notes
    -----
    - Spatial dimensions must be powers of 2 for wavelet decomposition
    - For 3D, only first two dimensions are decomposed (time preserved)
    - ReLU activation applied between MWNO layers (except after last layer)
    - Output channel dimension auto-squeezed if out_channels=1

    Input Shapes:
        - 1D: (batch, n_points, in_channels) where n_points must be a power of 2
        - 2D: (batch, height, width, in_channels) where height, width must be powers of 2
        - 3D: (batch, height, width, time, in_channels) where height, width must be powers of 2

    Output Shapes:
        Same spatial dimensions as input, with out_channels feature dimension.
        If out_channels=1, the channel dimension is squeezed.

    Examples
    --------

    >>> from neuralop.models import MWNO
    >>> # 1D time series operator
    >>> model_1d = MWNO(alpha=12, n_dim=1, in_channels=1, out_channels=1)
    >>> x = torch.randn(32, 256, 1)  # (batch, time, channels)
    >>> y = model_1d(x)  # (32, 256) - channel squeezed

    >>> # 2D image-to-image operator
    >>> model_2d = MWNO(n_modes=(16, 16), in_channels=3, out_channels=1)
    >>> x = torch.randn(16, 64, 64, 3)  # (batch, H, W, channels)
    >>> y = model_2d(x)  # (16, 64, 64) - channel squeezed

    References
    ----------
    .. [1] :

    Gupta, G., Xiao, X. and Bogdan, P., 2021.
        Multiwavelet-based operator learning for differential equations.
        Advances in Neural Information Processing Systems, 34, pp.24048-24062.

    @article{gupta2021multiwavelet,
        title={Multiwavelet-based operator learning for differential equations},
        author={Gupta, Gaurav and Xiao, Xiongye and Bogdan, Paul},
        journal={Advances in neural information processing systems},
        volume={34},
        pages={24048--24062},
        year={2021}
    }

    .. [2] :

    Xiao, Xiongye, et al. "Coupled multiwavelet operator learning for coupled differential equations."
        The Eleventh International Conference on Learning Representations. 2022.

    @inproceedings{xiao2023coupled,
    title={Coupled Multiwavelet Operator Learning for Coupled Differential Equations},
    author={Xiongye Xiao and Defu Cao and Ruochen Yang and Gaurav Gupta and Gengshuo Liu and Chenzhong Yin and Radu Balan and Paul Bogdan},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=kIo_C6QmMOM}

    """

    # Default hidden dimension when projection_channels=0
    DEFAULT_PROJECTION_HIDDEN = 128

    def __init__(
            self,
            n_modes: Optional[Union[int, Tuple[int, ...]]] = None,
            alpha: Optional[int] = None,
            n_dim: Optional[int] = None,
            in_channels: int = 1,
            out_channels: int = 1,
            k: int = 3,
            c: int = 1,
            n_layers: int = 3,
            L: int = 0,
            lifting_channels: int = 0,
            projection_channels: int = 128,
            base: str = 'legendre',
            initializer: Optional[Callable] = None,
            **kwargs
    ):
        super().__init__()

        # Parse and validate mode specification
        self.n_modes, self.n_dim, alpha = self._parse_mode_specification(
            n_modes, alpha, n_dim
        )

        # Store configuration
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.c = c
        self.n_layers = n_layers
        self.L = L
        self.base = base
        self.alpha = alpha

        # Calculate channel multiplier for wavelet representation
        # 1D: c * k coefficients, 2D/3D: c * k^2 coefficients
        if self.n_dim == 1:
            channel_multiplier = c * k
        else:
            channel_multiplier = c * k ** 2

        self.channel_multiplier = channel_multiplier

        # Build network layers
        self.lifting = self._build_lifting_layer(
            in_channels, channel_multiplier, lifting_channels
        )

        self.mwno_layers = nn.ModuleList([
            MWNO_CZ(
                k=k,
                alpha=alpha,
                L=L,
                c=c,
                base=base,
                n_dim=self.n_dim,
                initializer=initializer
            )
            for _ in range(n_layers)
        ])

        self.projection = self._build_projection_layer(
            channel_multiplier, out_channels, projection_channels
        )

        # Apply custom initialization if provided
        if initializer is not None:
            self._reset_parameters(initializer)

    def _parse_mode_specification(
            self,
            n_modes: Optional[Union[int, Tuple[int, ...]]],
            alpha: Optional[int],
            n_dim: Optional[int]
    ) -> Tuple[Tuple[int, ...], int, int]:
        """
        Parse and validate the mode specification parameters.

        Supports two ways to specify modes:
        1. n_modes as tuple: directly specifies modes per dimension
        2. alpha + n_dim: uses same alpha for all dimensions

        Parameters
        ----------
        n_modes : int or Tuple[int, ...], optional
            Direct mode specification
        alpha : int, optional
            Uniform modes for all dimensions
        n_dim : int, optional
            Number of spatial dimensions

        Returns
        -------
        n_modes : Tuple[int, ...]
            Parsed modes per dimension
        n_dim : int
            Number of spatial dimensions
        alpha : int
            Mode value (first element of n_modes)

        Raises
        ------
        ValueError
            If specification is invalid or ambiguous
        """
        if n_modes is not None:
            if isinstance(n_modes, (tuple, list)):
                # n_modes is tuple: infer n_dim from length
                parsed_modes = tuple(n_modes)
                parsed_n_dim = len(parsed_modes)
                parsed_alpha = n_modes[0]
            else:
                # n_modes is int: assume 1D
                parsed_alpha = n_modes
                parsed_n_dim = 1
                parsed_modes = (parsed_alpha,)

        elif alpha is not None:
            # Use alpha + n_dim specification
            if n_dim is None:
                parsed_n_dim = 1  # Default to 1D
            else:
                parsed_n_dim = n_dim
            parsed_modes = tuple([alpha] * parsed_n_dim)
            parsed_alpha = alpha

        else:
            raise ValueError(
                "Either 'n_modes' or 'alpha' must be specified. "
                "Examples:\n"
                "  - MWNO(n_modes=(12, 12), ...)\n"
                "  - MWNO(alpha=12, n_dim=2, ...)"
            )

        # Validate dimensionality
        if parsed_n_dim not in [1, 2, 3]:
            raise ValueError(
                f"MWNO only supports 1D, 2D, and 3D. Got {parsed_n_dim}D.\n"
                f"Parsed n_modes: {parsed_modes}"
            )

        return parsed_modes, parsed_n_dim, parsed_alpha

    def _build_lifting_layer(
            self,
            in_channels: int,
            channel_multiplier: int,
            lifting_channels: int
    ) -> nn.Module:
        """
        Build the lifting layer that embeds inputs into wavelet space.

        The lifting layer transforms from physical input space to the
        higher-dimensional wavelet coefficient space where MWNO operates.

        Parameters
        ----------
        in_channels : int
            Input feature dimension
        channel_multiplier : int
            Output dimension (c * k^n_dim)
        lifting_channels : int
            Hidden dimension (0 for direct linear projection)

        Returns
        -------
        nn.Module
            Lifting layer (Linear or Sequential MLP)
        """
        if lifting_channels > 0:
            # Two-layer MLP with nonlinearity
            return nn.Sequential(
                nn.Linear(in_channels, lifting_channels),
                nn.ReLU(inplace=True),
                nn.Linear(lifting_channels, channel_multiplier)
            )
        else:
            # Direct linear projection
            return nn.Linear(in_channels, channel_multiplier)

    def _build_projection_layer(
            self,
            channel_multiplier: int,
            out_channels: int,
            projection_channels: int
    ) -> nn.Module:
        """
        Build the projection layer that maps from wavelet space to output.

        The projection layer transforms from the wavelet coefficient space
        back to the physical output space.

        Parameters
        ----------
        channel_multiplier : int
            Input dimension (c * k^n_dim)
        out_channels : int
            Output feature dimension
        projection_channels : int
            Hidden dimension (0 uses default)

        Returns
        -------
        nn.Module
            Projection layer (two-layer MLP with ReLU)
        """
        if projection_channels > 0:
            hidden_dim = projection_channels
        else:
            hidden_dim = self.DEFAULT_PROJECTION_HIDDEN

        return nn.Sequential(
            nn.Linear(channel_multiplier, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_channels)
        )

    def _reset_parameters(self, initializer: Callable) -> None:
        """
        Apply custom initialization to all Linear layers.

        Parameters
        ----------
        initializer : callable
            Initialization function with signature: initializer(weight) -> None
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                initializer(module.weight)

    def _validate_input_shape(self, x: torch.Tensor) -> None:
        """
        Validate input tensor shape and dimensions.

        Checks:
        1. Number of dimensions matches expected (batch + spatial + channels)
        2. Channel dimension matches in_channels
        3. Spatial dimensions are powers of 2 (required for wavelets)

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to validate

        Raises
        ------
        AssertionError
            If any validation check fails
        """
        # Check number of dimensions
        expected_ndim = self.n_dim + 2  # batch + spatial_dims + channel
        assert x.ndim == expected_ndim, (
            f"Expected {expected_ndim}D input for {self.n_dim}D MWNO "
            f"(batch + {self.n_dim} spatial + channel), "
            f"but got {x.ndim}D tensor with shape {x.shape}"
        )

        # Check channel dimension
        assert x.shape[-1] == self.in_channels, (
            f"Expected {self.in_channels} input channels, "
            f"but got {x.shape[-1]}. Input shape: {x.shape}"
        )

        # Check spatial dimensions are powers of 2
        if self.n_dim == 1:
            n_points = x.shape[1]
            assert self._is_power_of_2(n_points), (
                f"Spatial dimension must be power of 2, got n_points={n_points}. "
                f"Valid sizes: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, ..."
            )

        elif self.n_dim == 2:
            height, width = x.shape[1], x.shape[2]
            assert self._is_power_of_2(height) and self._is_power_of_2(width), (
                f"Spatial dimensions must be powers of 2, got height={height}, width={width}. "
                f"Valid sizes: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, ..."
            )

        elif self.n_dim == 3:
            height, width = x.shape[1], x.shape[2]
            # For 3D, only first two spatial dims need to be powers of 2
            assert self._is_power_of_2(height) and self._is_power_of_2(width), (
                f"Spatial dimensions (height, width) must be powers of 2 for 3D MWNO, "
                f"got height={height}, width={width}. Time dimension can be arbitrary. "
                f"Valid sizes: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, ..."
            )

    @staticmethod
    def _is_power_of_2(n: int) -> bool:
        """Check if n is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0

    def _reshape_to_wavelet_format(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape lifted features to wavelet coefficient format.

        Adds the wavelet basis dimension by splitting the channel dimension
        into (wavelet_channels, basis_functions).

        Parameters
        ----------
        x : torch.Tensor
            Lifted features with shape (*spatial, c*k^n_dim)

        Returns
        -------
        torch.Tensor
            Wavelet format:
            - 1D: (batch, n_points, c, k)
            - 2D: (batch, height, width, c, k²)
            - 3D: (batch, height, width, time, c, k²)
        """
        if self.n_dim == 1:
            batch_size, n_points, lifted_channels = x.shape
            expected_channels = self.c * self.k
            assert lifted_channels == expected_channels, (
                f"Lifting produced {lifted_channels} channels, "
                f"expected {expected_channels} (c={self.c} × k={self.k})"
            )
            return x.view(batch_size, n_points, self.c, self.k)

        elif self.n_dim == 2:
            batch_size, height, width, lifted_channels = x.shape
            expected_channels = self.c * self.k ** 2
            assert lifted_channels == expected_channels, (
                f"Lifting produced {lifted_channels} channels, "
                f"expected {expected_channels} (c={self.c} × k²={self.k ** 2})"
            )
            return x.view(batch_size, height, width, self.c, self.k ** 2)

        elif self.n_dim == 3:
            batch_size, height, width, time_steps, lifted_channels = x.shape
            expected_channels = self.c * self.k ** 2
            assert lifted_channels == expected_channels, (
                f"Lifting produced {lifted_channels} channels, "
                f"expected {expected_channels} (c={self.c} × k²={self.k ** 2})"
            )
            return x.view(batch_size, height, width, time_steps, self.c, self.k ** 2)

    def _reshape_from_wavelet_format(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape from wavelet coefficient format back to standard format.

        Flattens the (wavelet_channels, basis_functions) dimensions back
        into a single channel dimension.

        Parameters
        ----------
        x : torch.Tensor
            Wavelet format (see _reshape_to_wavelet_format)

        Returns
        -------
        torch.Tensor
            Standard format: (*spatial, c*k^n_dim)
        """
        if self.n_dim == 1:
            batch_size, n_points = x.shape[0], x.shape[1]
            return x.view(batch_size, n_points, -1)

        elif self.n_dim == 2:
            batch_size, height, width = x.shape[0], x.shape[1], x.shape[2]
            return x.view(batch_size, height, width, -1)

        elif self.n_dim == 3:
            batch_size, height, width, time_steps = (
                x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            )
            return x.view(batch_size, height, width, time_steps, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Multiwavelet Neural Operator.

        Processing pipeline:
        1. Validate input shape and dimensions
        2. Lift to wavelet space: input → high-dimensional wavelet coefficients
        3. Apply MWNO layers with ReLU activation (except last layer)
        4. Project back to output space: wavelet coefficients → output
        5. Squeeze channel dimension if out_channels=1

        Parameters
        ----------
        x : torch.Tensor
            Input function values
            - 1D: (batch, n_points, in_channels)
            - 2D: (batch, height, width, in_channels)
            - 3D: (batch, height, width, time, in_channels)

            Requirements:
            - Spatial dimensions must be powers of 2
            - n_points, height, width ∈ {2, 4, 8, 16, 32, 64, 128, ...}

        Returns
        -------
        torch.Tensor
            Output function values
            - 1D: (batch, n_points, out_channels) or (batch, n_points) if out_channels=1
            - 2D: (batch, height, width, out_channels) or (batch, height, width) if out_channels=1
            - 3D: (batch, height, width, time, out_channels) or (batch, height, width, time) if out_channels=1

        Examples
        --------
        >>> model = MWNO(alpha=12, n_dim=2, in_channels=1, out_channels=1)
        >>> x = torch.randn(16, 64, 64, 1)
        >>> y = model(x)  # shape: (16, 64, 64) - channel squeezed
        """
        # Validate input
        self._validate_input_shape(x)

        # Lift to wavelet space
        x = self.lifting(x)

        # Reshape to wavelet coefficient format
        x = self._reshape_to_wavelet_format(x)

        # Apply MWNO transformation layers
        for layer_idx, layer in enumerate(self.mwno_layers):
            x = layer(x)
            # Apply ReLU between layers (but not after last layer)
            if layer_idx < self.n_layers - 1:
                x = F.relu(x)

        # Reshape back to standard format
        x = self._reshape_from_wavelet_format(x)

        # Project to output space
        x = self.projection(x)

        # Validate output channels
        assert x.shape[-1] == self.out_channels, (
            f"Expected {self.out_channels} output channels, "
            f"but got {x.shape[-1]}. This is an internal error."
        )

        # Squeeze channel dimension for scalar outputs
        if self.out_channels == 1:
            return x.squeeze(-1)

        return x