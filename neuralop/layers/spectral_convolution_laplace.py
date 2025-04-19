import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Sequence
from typing import List, Optional, Tuple, Union
from .resample import resample
from ..utils import validate_scaling_factor
from .shape_enforcer import ShapeEnforcer
from .base_spectral_conv import BaseSpectralConv

Number = Union[int, float]


def _compute_dt(shape, start_points: List = None, end_points: List = None):
    """
    Compute uniform spacing (dt) for each dimension based on domain lengths, step sizes,
    start points, and end points. Defaults to a unit domain if not specified.

    Parameters:
    ----------
    shape (Sequence[int]): The shape of the input excluding batch and channel, i.e. (d_1, d_2, ..., d_n).
    start_points (Sequence[float], optional): Start points for each dimension. Defaults to 0.0 for all dimensions.
    end_points (Sequence[float], optional): End points for each dimension. Defaults to 1.0 for all dimensions.

    Returns:
    -------
    dt_list (Sequence[float]): A list of spacings, one per dimension.
    grid (List[torch.Tensor]): A list of grid points for each dimension based on the spacing and domain.
    """
    dim = len(shape)
    # Set default start and end points if not provided
    if start_points is None:
        start_points = torch.zeros(dim).tolist()
    if end_points is None:
        end_points = torch.ones(dim).tolist()

    # Validate that start_points and end_points match the number of dimensions
    if len(start_points) != dim or len(end_points) != dim:
        raise ValueError(
            "Start points and end points must match the number of input dimensions ({dim})."
        )

    # Compute domain lengths from start and end points
    domain_lengths = [end_points[i] - start_points[i] for i in range(dim)]

    # Generate grid points for each dimension using torch.linspace
    grid = [
        torch.linspace(start_points[i], end_points[i], steps=shape[i])
        for i in range(dim)
    ]

    # Compute dt directly from the grid
    dt_list = [(grid[i][1] - grid[i][0]).item() for i in range(dim)]

    return dt_list, grid


# ====================================
#  Laplace layer: pole-residue operation is used to calculate the poles and residues of the output
# ====================================
class SpectralConvLaplace1D(BaseSpectralConv):
    """Implements 1D Laplace-based Spectral Convolution.

    This class implements a spectral convolution layer using pole-residue formulation
    for 1D data. Instead of using standard Fourier transforms, it uses pole-residue
    operations to calculate the response in the frequency domain.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_modes : int or int tuple
        Number of modes to use for pole-residue calculation
    complex_data : bool, optional
        whether data takes on complex values in the spatial domain, by default False
    max_n_modes : int tuple or None, default is None
        Maximum number of modes to keep in the Laplace layer
    bias : bool, default is True
        whether to include bias term
    separable : bool, default is False
        whether to use separable implementation
    resolution_scaling_factor : float or float list, optional
        Factor to scale the output resolution
    xno_block_precision : str, default is "full"
        Precision to use for the convolution block
    rank : float, optional
        Rank parameter, by default 0.5
    factorization : str or None, optional
        Type of factorization to use
    implementation : str, default is "reconstructed"
        Implementation approach for the convolution
    fixed_rank_modes : bool, default is False
        Whether to use fixed rank modes
    decomposition_kwargs : dict, optional
        Additional parameters for decomposition
    init_std : str or float, default is "auto"
        Standard deviation for weight initialization
    fft_norm : str, default is "forward"
        Normalization method for FFT
    device : torch.device, optional
        Device to use for computation
    linspace_steps : list, optional
        Number of steps for each dimension
    linspace_startpoints : list, optional
        Start points for each dimension
    linspace_endpoints : list, optional
        End points for each dimension
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        complex_data=False,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        xno_block_precision="full",
        rank=0.5,
        factorization=None,
        implementation="reconstructed",
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="forward",
        device=None,
        linspace_steps=None,
        linspace_startpoints=None,
        linspace_endpoints=None,
    ):
        super(SpectralConvLaplace1D, self).__init__(device=device)

        # Store grid parameters for domain specification
        self.linspace_steps = linspace_steps
        self.linspace_startpoints = linspace_startpoints
        self.linspace_endpoints = linspace_endpoints
        self.n_modes = n_modes

        self.order = len(self.n_modes)
        self.resolution_scaling_factor: Union[None, List[List[float]]] = (
            validate_scaling_factor(resolution_scaling_factor, self.order)
        )

        # Set up and initialize weight parameters
        (max_n_modes,) = self.n_modes
        self.max_n_modes = self.n_modes

        self.scale = 1 / (in_channels * out_channels)

        # Initialize single weight tensor combining poles and residues
        # Weight format: [in_channels, out_channels, poles + residues]
        total_modes = max_n_modes + 1 * max_n_modes
        self.weight = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                total_modes,
                dtype=torch.cfloat,
            )
        )

        self.shape_enforcer = ShapeEnforcer()

    def transform(self, x, output_shape=None):
        """Transform the input tensor to match desired output shape.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        output_shape : tuple, optional
            Desired output spatial dimensions, by default None

        Returns
        -------
        torch.Tensor
            Transformed tensor
        """
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [
                    round(s * r)
                    for (s, r) in zip(in_shape, self.resolution_scaling_factor)
                ]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    def output_PR(self, lambda1, alpha, weights_pole, weights_residue):
        """Calculate output using pole-residue formulation.

        Parameters
        ----------
        lambda1 : torch.Tensor
            Frequency domain grid points
        alpha : torch.Tensor
            FFT of input
        weights_pole : torch.Tensor
            Pole weights
        weights_residue : torch.Tensor
            Residue weights

        Returns
        -------
        tuple
            Tuple of (transient part, steady-state part)
        """
        # Initialize output tensor for frequency response
        Hw = torch.zeros(
            weights_residue.shape[0],
            weights_residue.shape[0],
            weights_residue.shape[2],
            lambda1.shape[0],
            device=alpha.device,
            dtype=torch.cfloat,
        )

        # Calculate pole-residue representation in the Laplace domain
        term1 = torch.div(1, torch.sub(lambda1, weights_pole))
        Hw = weights_residue * term1

        # Compute both transient and steady-state parts
        output_residue1 = torch.einsum("bix,xiok->box", alpha, Hw)
        output_residue2 = torch.einsum("bix,xiok->bok", alpha, -Hw)

        return output_residue1, output_residue2

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        """Forward pass for 1D Laplace spectral convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, L)
        output_shape : tuple, optional
            Desired output spatial dimensions, by default None

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, L) or output_shape
        """
        batchsize, channels, *mode_sizes = x.shape

        # Setup computation parameters
        (modes1,) = self.n_modes
        L = x.shape[-1]
        modes1 = min(modes1, L)
        self.linspace_steps = x.shape[2:]

        # Compute the grid and spacing for the domain
        dt_list, shape = _compute_dt(
            shape=self.linspace_steps,
            start_points=self.linspace_startpoints,
            end_points=self.linspace_endpoints,
        )
        t = shape[0].to(x.device)
        dt = dt_list[0]

        # Transform to frequency domain and compute frequency grid
        alpha = torch.fft.fft(x, dim=-1)
        lambda0 = (
            torch.fft.fftfreq(t.shape[0], dt, device=alpha.device) * 2 * np.pi * 1j
        )
        lambda1 = lambda0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Extract pole and residue weights from the combined weight tensor
        weights_pole = self.weight[:, :, :modes1].view(
            self.weight.size(0), self.weight.size(1), modes1
        )
        weights_residue = self.weight[:, :, modes1 : (modes1 * 2)].view(
            self.weight.size(0), self.weight.size(1), modes1
        )

        # Calculate frequency response using pole-residue formulation
        output_residue1, output_residue2 = self.output_PR(
            lambda1, alpha, weights_pole, weights_residue
        )

        # Convert transient part back to time domain
        x1 = torch.fft.ifft(output_residue1, n=x.size(-1))
        x1 = torch.real(x1)

        # Calculate steady-state response in time domain
        x2 = torch.zeros(
            output_residue2.shape[0],
            output_residue2.shape[1],
            t.shape[0],
            device=alpha.device,
            dtype=torch.cfloat,
        )
        # Apply exponential terms for time-domain calculation
        term1 = torch.einsum(
            "iok,az->iokz", weights_pole, t.type(torch.complex64).reshape(1, -1)
        )
        term2 = torch.exp(term1)
        x2 = torch.einsum("bok,iokz->boz", output_residue2, term2)
        x2 = torch.real(x2) / x.size(-1)

        # Combine transient and steady-state responses
        x = x1 + x2

        # Adjust output shape if needed
        if self.resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple(
                [
                    round(s * r)
                    for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)
                ]
            )
        if output_shape is not None:
            mode_sizes = output_shape

        # Ensure the output has the correct shape
        if list(x.shape[2:]) != mode_sizes:
            x = self.shape_enforcer(x, output_shape=mode_sizes)

        return x


class SpectralConvLaplace2D(BaseSpectralConv):
    """Implements 2D Laplace-based Spectral Convolution.

    This class implements a spectral convolution layer using pole-residue formulation
    for 2D data. It extends the concepts of the 1D version to handle 2D spatial domains.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_modes : tuple
        Number of modes to use for pole-residue calculation (modes1, modes2)
    complex_data : bool, optional
        whether data takes on complex values in the spatial domain, by default False
    max_n_modes : tuple or None, default is None
        Maximum number of modes to keep in the Laplace layer
    bias : bool, default is True
        whether to include bias term
    separable : bool, default is False
        whether to use separable implementation
    resolution_scaling_factor : float or float list, optional
        Factor to scale the output resolution
    xno_block_precision : str, default is "full"
        Precision to use for the convolution block
    rank : float, optional
        Rank parameter, by default 0.5
    factorization : str or None, optional
        Type of factorization to use
    implementation : str, default is "reconstructed"
        Implementation approach for the convolution
    fixed_rank_modes : bool, default is False
        Whether to use fixed rank modes
    decomposition_kwargs : dict, optional
        Additional parameters for decomposition
    init_std : str or float, default is "auto"
        Standard deviation for weight initialization
    fft_norm : str, default is "forward"
        Normalization method for FFT
    device : torch.device, optional
        Device to use for computation
    linspace_steps : list, optional
        Number of steps for each dimension
    linspace_startpoints : list, optional
        Start points for each dimension
    linspace_endpoints : list, optional
        End points for each dimension
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        complex_data=False,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        xno_block_precision="full",
        rank=0.5,
        factorization=None,
        implementation="reconstructed",
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="forward",
        device=None,
        linspace_steps=None,
        linspace_startpoints=None,
        linspace_endpoints=None,
    ):
        super(SpectralConvLaplace2D, self).__init__(device=device)

        # Store grid parameters for domain specification
        self.linspace_steps = linspace_steps
        self.linspace_startpoints = linspace_startpoints
        self.linspace_endpoints = linspace_endpoints
        self.n_modes = n_modes

        self.order = len(self.n_modes)
        self.resolution_scaling_factor: Union[None, List[List[float]]] = (
            validate_scaling_factor(resolution_scaling_factor, self.order)
        )

        # Set up dimensions and weight parameters
        max_modes1, max_modes2 = self.n_modes
        self.max_n_modes = self.n_modes
        self.scale = 1 / (in_channels * out_channels)

        # Initialize single weight tensor combining poles and residues
        # Shape: (in_channels, out_channels, modes1 + modes2 + modes1 * modes2)
        # The tensor contains weights for poles in dimension 1, poles in dimension 2,
        # and residues for all combinations of poles
        total_modes = max_modes1 + max_modes2 + (max_modes1 * max_modes2)
        self.weight = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, total_modes, dtype=torch.cfloat)
        )

        self.shape_enforcer = ShapeEnforcer()

    def transform(self, x, output_shape=None):
        """Transform the input tensor to match desired output shape.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        output_shape : tuple, optional
            Desired output spatial dimensions, by default None

        Returns
        -------
        torch.Tensor
            Transformed tensor
        """
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [
                    round(s * r)
                    for (s, r) in zip(in_shape, self.resolution_scaling_factor)
                ]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    def output_PR(
        self, lambda1, lambda2, alpha, weights_pole1, weights_pole2, weights_residue
    ):
        """Calculate output using pole-residue formulation for 2D.

        Parameters
        ----------
        lambda1 : torch.Tensor
            Frequency domain grid points for first dimension
        lambda2 : torch.Tensor
            Frequency domain grid points for second dimension
        alpha : torch.Tensor
            FFT of input
        weights_pole1 : torch.Tensor
            Pole weights for first dimension
        weights_pole2 : torch.Tensor
            Pole weights for second dimension
        weights_residue : torch.Tensor
            Residue weights

        Returns
        -------
        tuple
            Tuple of (transient part, steady-state part)
        """
        # Initialize output tensor for frequency response
        Hw = torch.zeros(
            weights_residue.shape[0],
            weights_residue.shape[0],
            weights_residue.shape[2],
            weights_residue.shape[3],
            lambda1.shape[0],
            lambda2.shape[0],
            device=alpha.device,
            dtype=torch.cfloat,
        )

        # Calculate pole-residue representation in the 2D Laplace domain
        # by combining poles from both dimensions
        term1 = torch.div(
            1,
            torch.einsum(
                "pbix,qbik->pqbixk",
                torch.sub(lambda1, weights_pole1),
                torch.sub(lambda2, weights_pole2),
            ),
        )
        Hw = torch.einsum("bixk,pqbixk->pqbixk", weights_residue, term1)

        # Note: Different PDEs use different signs for the steady-state part
        Pk = Hw  # for ode, Pk=-Hw; for 2d pde, Pk=Hw; for 3d pde, Pk=-Hw;

        # Compute both transient and steady-state parts
        output_residue1 = torch.einsum("biox,oxikpq->bkox", alpha, Hw)
        output_residue2 = torch.einsum("biox,oxikpq->bkpq", alpha, Pk)
        return output_residue1, output_residue2

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        """Forward pass for 2D Laplace spectral convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, H, W)
        output_shape : tuple, optional
            Desired output spatial dimensions, by default None

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, H, W) or output_shape
        """
        batchsize, channels, *mode_sizes = x.shape

        # Setup computation parameters
        modes1, modes2 = self.n_modes
        H, W = x.shape[-2], x.shape[-1]
        modes1, modes2 = min(modes1, H), min(modes2, W)
        self.linspace_steps = x.shape[2:]

        # Compute the grid and spacing for the domain
        dt_list, shape = _compute_dt(
            shape=self.linspace_steps,
            start_points=self.linspace_startpoints,
            end_points=self.linspace_endpoints,
        )
        ty = shape[0].to(x.device)
        tx = shape[1].to(x.device)
        dty = dt_list[0]
        dtx = dt_list[1]

        # Transform to frequency domain and compute frequency grids
        alpha = torch.fft.fft2(x, dim=[-2, -1])
        omega1 = (
            torch.fft.fftfreq(ty.shape[0], dty, device=alpha.device) * 2 * np.pi * 1j
        )
        omega2 = (
            torch.fft.fftfreq(tx.shape[0], dtx, device=alpha.device) * 2 * np.pi * 1j
        )

        # Prepare frequency grids for computation
        omega1 = omega1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        omega2 = omega2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1, lambda2 = omega1, omega2

        # Extract pole and residue weights from the combined weight tensor
        weights_pole1 = self.weight[:, :, :modes1].view(
            self.weight.size(0), self.weight.size(1), modes1
        )
        weights_pole2 = self.weight[:, :, modes1 : (modes1 + modes2)].view(
            self.weight.size(0), self.weight.size(1), modes2
        )
        weights_residue = self.weight[
            :, :, (modes1 + modes2) : (modes1 + modes2 + modes1 * modes2)
        ].view(self.weight.size(0), self.weight.size(1), modes1, modes2)

        # Calculate frequency response using pole-residue formulation
        output_residue1, output_residue2 = self.output_PR(
            lambda1, lambda2, alpha, weights_pole1, weights_pole2, weights_residue
        )

        # Convert transient part back to spatial domain
        x1 = torch.fft.ifft2(output_residue1, s=(x.size(-2), x.size(-1)))
        x1 = torch.real(x1)

        # Calculate steady-state response in spatial domain
        # Create exponential terms for each dimension
        term1 = torch.einsum("iop,z->iopz", weights_pole1, ty.type(torch.complex64))
        term2 = torch.einsum("ioq,x->ioqx", weights_pole2, tx.type(torch.complex64))
        term1 = torch.exp(term1)
        term2 = torch.exp(term2)

        # Combine exponential terms across dimensions
        term3 = torch.einsum("iopz,ioqx->iopqzx", term1, term2)

        # Apply steady-state response
        x2 = torch.einsum("bopq,iopqzx->bozx", output_residue2, term3)
        x2 = torch.real(x2) / (x.size(-1) * x.size(-2))

        # Combine transient and steady-state responses
        x = x1 + x2

        # Adjust output shape if needed
        if self.resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple(
                [
                    round(s * r)
                    for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)
                ]
            )
        if output_shape is not None:
            mode_sizes = output_shape

        # Ensure the output has the correct shape
        if list(x.shape[2:]) != mode_sizes:
            x = self.shape_enforcer(x, output_shape=mode_sizes)

        return x


class SpectralConvLaplace3D(BaseSpectralConv):
    """Implements 3D Laplace-based Spectral Convolution.

    This class implements a spectral convolution layer using pole-residue formulation
    for 3D data. It extends the concepts of the 1D and 2D versions to handle 3D spatial domains.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_modes : tuple
        Number of modes to use for pole-residue calculation (modes1, modes2, modes3)
    complex_data : bool, optional
        whether data takes on complex values in the spatial domain, by default False
    max_n_modes : tuple or None, default is None
        Maximum number of modes to keep in the Laplace layer
    bias : bool, default is True
        whether to include bias term
    separable : bool, default is False
        whether to use separable implementation
    resolution_scaling_factor : float or float list, optional
        Factor to scale the output resolution
    xno_block_precision : str, default is "full"
        Precision to use for the convolution block
    rank : float, optional
        Rank parameter, by default 0.5
    factorization : str or None, optional
        Type of factorization to use
    implementation : str, default is "reconstructed"
        Implementation approach for the convolution
    fixed_rank_modes : bool, default is False
        Whether to use fixed rank modes
    decomposition_kwargs : dict, optional
        Additional parameters for decomposition
    init_std : str or float, default is "auto"
        Standard deviation for weight initialization
    fft_norm : str, default is "forward"
        Normalization method for FFT
    device : torch.device, optional
        Device to use for computation
    linspace_steps : list, optional
        Number of steps for each dimension
    linspace_startpoints : list, optional
        Start points for each dimension
    linspace_endpoints : list, optional
        End points for each dimension
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        complex_data=False,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        xno_block_precision="full",
        rank=0.5,
        factorization=None,
        implementation="reconstructed",
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="forward",
        device=None,
        linspace_steps=None,
        linspace_startpoints=None,
        linspace_endpoints=None,
    ):
        super(SpectralConvLaplace3D, self).__init__(device=device)

        # Store grid parameters for domain specification
        self.linspace_steps = linspace_steps
        self.linspace_startpoints = linspace_startpoints
        self.linspace_endpoints = linspace_endpoints
        self.n_modes = n_modes

        self.order = len(self.n_modes)
        self.resolution_scaling_factor: Union[None, List[List[float]]] = (
            validate_scaling_factor(resolution_scaling_factor, self.order)
        )

        # Set up dimensions and weight parameters
        self.modes1, self.modes2, self.modes3 = self.n_modes
        max_modes1, max_modes2, max_modes3 = self.n_modes
        self.max_n_modes = self.n_modes

        # Compute total number of modes to combine into single weight tensor
        # Format: [poles for dim1] + [poles for dim2] + [poles for dim3] + [residues for all combinations]
        total_modes = (
            max_modes1
            + max_modes2
            + max_modes3
            + (max_modes1 * max_modes2 * max_modes3)
        )

        self.scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, total_modes, dtype=torch.cfloat)
        )

        self.shape_enforcer = ShapeEnforcer()

    def transform(self, x, output_shape=None):
        """Transform the input tensor to match desired output shape.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        output_shape : tuple, optional
            Desired output spatial dimensions, by default None

        Returns
        -------
        torch.Tensor
            Transformed tensor
        """
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [
                    round(s * r)
                    for (s, r) in zip(in_shape, self.resolution_scaling_factor)
                ]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    def output_PR(
        self,
        lambda1,
        lambda2,
        lambda3,
        alpha,
        weights_pole1,
        weights_pole2,
        weights_pole3,
        weights_residue,
    ):
        """Calculate output using pole-residue formulation for 3D.

        Parameters
        ----------
        lambda1 : torch.Tensor
            Frequency domain grid points for first dimension
        lambda2 : torch.Tensor
            Frequency domain grid points for second dimension
        lambda3 : torch.Tensor
            Frequency domain grid points for third dimension
        alpha : torch.Tensor
            FFT of input
        weights_pole1 : torch.Tensor
            Pole weights for first dimension
        weights_pole2 : torch.Tensor
            Pole weights for second dimension
        weights_pole3 : torch.Tensor
            Pole weights for third dimension
        weights_residue : torch.Tensor
            Residue weights

        Returns
        -------
        tuple
            Tuple of (transient part, steady-state part)
        """
        # Initialize output tensor for frequency response
        Hw = torch.zeros(
            weights_residue.shape[0],
            weights_residue.shape[0],
            weights_residue.shape[2],
            weights_residue.shape[3],
            weights_residue.shape[4],
            lambda1.shape[0],
            lambda2.shape[0],
            lambda2.shape[3],
            device=alpha.device,
            dtype=torch.cfloat,
        )

        # Calculate pole-residue representation in the 3D Laplace domain
        # by combining poles from all three dimensions
        term1 = torch.div(
            1,
            torch.einsum(
                "pbix,qbik,rbio->pqrbixko",
                torch.sub(lambda1, weights_pole1),
                torch.sub(lambda2, weights_pole2),
                torch.sub(lambda3, weights_pole3),
            ),
        )
        Hw = torch.einsum("bixko,pqrbixko->pqrbixko", weights_residue, term1)

        # Compute both transient and steady-state parts
        # Note: For 3D PDEs, the steady-state part uses negative Hw
        output_residue1 = torch.einsum("bioxs,oxsikpqr->bkoxs", alpha, Hw)
        output_residue2 = torch.einsum("bioxs,oxsikpqr->bkpqr", alpha, -Hw)
        return output_residue1, output_residue2

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        """Forward pass for 3D Laplace spectral convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, D, H, W)
        output_shape : tuple, optional
            Desired output spatial dimensions, by default None

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, D, H, W) or output_shape
        """
        batchsize, channels, *mode_sizes = x.shape

        # Setup computation parameters
        modes1, modes2, modes3 = self.n_modes
        D, H, W = x.shape[-3], x.shape[-2], x.shape[-1]
        modes1, modes2, modes3 = min(modes1, D), min(modes2, H), min(modes3, W)
        self.linspace_steps = x.shape[2:]

        # Compute the grid and spacing for the domain
        dt_list, shape = _compute_dt(
            shape=self.linspace_steps,
            start_points=self.linspace_startpoints,
            end_points=self.linspace_endpoints,
        )
        tz = shape[0].to(x.device)
        tx = shape[1].to(x.device)
        ty = shape[2].to(x.device)
        dtz = dt_list[0]  # this can be time dimension, instead of Z dimension
        dtx = dt_list[1]
        dty = dt_list[2]

        # Transform to frequency domain and compute frequency grids
        alpha = torch.fft.fftn(x, dim=[-3, -2, -1])
        omega1 = (
            torch.fft.fftfreq(tz.shape[0], dtz, device=alpha.device) * 2 * np.pi * 1j
        )
        omega2 = (
            torch.fft.fftfreq(tx.shape[0], dtx, device=alpha.device) * 2 * np.pi * 1j
        )
        omega3 = (
            torch.fft.fftfreq(ty.shape[0], dty, device=alpha.device) * 2 * np.pi * 1j
        )

        # Slice frequencies to match the chosen modes
        omega1 = omega1[:modes1]
        omega2 = omega2[:modes2]
        omega3 = omega3[:modes3]

        # Prepare frequency grids for computation
        omega1 = omega1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        omega2 = omega2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        omega3 = omega3.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1, lambda2, lambda3 = omega1, omega2, omega3

        # Slice alpha to only consider the selected modes
        alpha = alpha[:, :, :modes1, :modes2, :modes3]

        # Extract pole and residue weights from the combined weight tensor
        i, o, _ = self.weight.shape
        weights_pole1 = self.weight[:, :, :modes1].view(i, o, modes1)
        weights_pole2 = self.weight[:, :, modes1 : (modes1 + modes2)].view(i, o, modes2)
        weights_pole3 = self.weight[
            :, :, (modes1 + modes2) : (modes1 + modes2 + modes3)
        ].view(i, o, modes3)
        weights_residue = self.weight[
            :,
            :,
            (modes1 + modes2 + modes3) : (
                modes1 + modes2 + modes3 + (modes1 * modes2 * modes3)
            ),
        ].view(i, o, modes1, modes2, modes3)

        # Calculate frequency response using pole-residue formulation
        output_residue1, output_residue2 = self.output_PR(
            lambda1,
            lambda2,
            lambda3,
            alpha,
            weights_pole1,
            weights_pole2,
            weights_pole3,
            weights_residue,
        )

        # Convert transient part back to spatial domain
        x1 = torch.fft.ifftn(output_residue1, s=(x.size(-3), x.size(-2), x.size(-1)))
        x1 = torch.real(x1)

        # Calculate steady-state response in spatial domain
        # Create exponential terms for each dimension
        term1 = torch.einsum(
            "bip,kz->bipz", weights_pole1, tz.type(torch.complex64).reshape(1, -1)
        )
        term2 = torch.einsum(
            "biq,kx->biqx", weights_pole2, tx.type(torch.complex64).reshape(1, -1)
        )
        term3 = torch.einsum(
            "bim,ky->bimy", weights_pole3, ty.type(torch.complex64).reshape(1, -1)
        )

        # Combine exponential terms across all three dimensions
        term4 = torch.einsum(
            "bipz,biqx,bimy->bipqmzxy",
            torch.exp(term1),
            torch.exp(term2),
            torch.exp(term3),
        )

        # Apply steady-state response
        x2 = torch.einsum("kbpqm,bipqmzxy->kizxy", output_residue2, term4)
        x2 = torch.real(x2) / x.size(-1) / x.size(-2) / x.size(-3)

        # Combine transient and steady-state responses
        x = x1 + x2

        # Adjust output shape if needed
        if self.resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple(
                [
                    round(s * r)
                    for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)
                ]
            )
        if output_shape is not None:
            mode_sizes = output_shape

        # Ensure the output has the correct shape
        if list(x.shape[2:]) != mode_sizes:
            x = self.shape_enforcer(x, output_shape=mode_sizes)

        return x
