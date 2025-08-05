import torch
import numpy as np
from scipy.special import poch

from .dct import idct, idct_2d, idct_3d


def get_fixed_coords(Ln1, Ln2):
    """
    Generates a flattened grid of coordinates.

    Args:
        Ln1 (int): Number of points in the first dimension.
        Ln2 (int): Number of points in the second dimension.

    Returns:
        torch.Tensor: A tensor of shape (Ln1*Ln2, 2) with grid coordinates.
    """
    xs = torch.linspace(0, 1, steps=Ln1 + 1)[0:-1]
    ys = torch.linspace(0, 1, steps=Ln2 + 1)[0:-1]
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    coords = torch.cat([yy.reshape(-1, 1), xx.reshape(-1, 1)], dim=-1)
    return coords


class GRFSampler(object):
    """Base class for Gaussian Random Field samplers."""

    def sample(self, N):
        raise NotImplementedError()


class RBFKernelSampler(GRFSampler):
    """
    Sampler for a Gaussian Random Field with a Radial Basis Function (RBF) kernel.
    This method uses a Cholesky decomposition of the covariance matrix. It can be
    memory-intensive for large grids.
    """

    @torch.no_grad()
    def __init__(self, in_channels, Ln1, Ln2, scale=1, eps=0.01, device=None):
        self.in_channels = in_channels
        self.Ln1 = Ln1
        self.Ln2 = Ln2
        self.device = device
        self.scale = scale

        # (Ln1*Ln2, 2)
        meshgrid = get_fixed_coords(self.Ln1, self.Ln2).to(device)
        # (Ln1*Ln2, Ln1*Ln2)
        C = torch.exp(-torch.cdist(meshgrid, meshgrid) / (2 * scale**2))

        # Add a small regularization term for numerical stability
        I = torch.eye(C.size(-1)).to(device)

        # Not memory efficient
        # C = C + (eps**2) * I
        I.mul_(eps**2)  # In-place multiply by eps**2
        C.add_(I)  # In-place add by I
        del I

        # Cholesky decomposition
        self.L = torch.linalg.cholesky(C)
        del C  # Free up memory

    @torch.no_grad()
    def sample(self, N):
        """
        Generates N samples from the GRF.

        Args:
            N (int): The number of samples to generate.

        Returns:
            torch.Tensor: A tensor of shape (N, in_channels, Ln1, Ln2).
        """
        samples = torch.zeros((N, self.Ln1 * self.Ln2, self.in_channels)).to(self.device)
        # Generate samples iteratively to save memory
        for ix in range(N):
            # (Ln1*Ln2, in_channels)
            this_z = torch.randn(self.Ln1 * self.Ln2, self.in_channels).to(self.device)
            # (Ln1*Ln2, Ln1*Ln2) @ (Ln1*Ln2, in_channels) -> (Ln1*Ln2, in_channels)
            samples[ix] = torch.matmul(self.L, this_z)

        # Reshape into (N, Ln1, Ln2, in_channels)
        sample_rshp = samples.reshape(-1, self.Ln1, self.Ln2, self.in_channels)

        # Transpose to (N, in_channels, Ln1, Ln2)
        sample_rshp = sample_rshp.permute(0, 3, 1, 2)

        return sample_rshp


class MaternKernelSampler(GRFSampler):
    @torch.no_grad()
    def __init__(
        self,
        in_channels,
        Ln1,
        Ln2,
        scale,
        nu,
        normalize_std=False,
        boundary_condition="none",
        device=None,
    ):
        """
        Gaussian random field sampler for a Matérn covariance function, implemented
        using spectral methods with PyTorch acceleration.

        The Matérn covariance function in the spatial domain is:
            C_ν(d) = σ² × (2^(1-ν)/Γ(ν)) × (√(2ν) × d/ρ)^ν × K_ν(√(2ν) × d/ρ)

        Where:
        - d is the distance between points
        - ρ (rho) is the length scale parameter
        - ν (nu) is the smoothness parameter
        - K_ν is the modified Bessel function of the second kind
        - σ² is the variance

        The spectral density in the frequency domain is:
            S(f) = σ² × (2^n × π^(n/2) × Γ(ν + n/2) × (2ν)^ν) / (Γ(ν) × ρ^(2ν)) × (2ν/ρ² + 4π²f²)^(-(ν + n/2))

        Args:
            in_channels (int): Number of input channels for the samples.
            Ln1, Ln2 (int): Grid dimensions.
            scale (float): The length scale `l` of the Matérn kernel.
                                 Controls the correlation distance.
            nu (float): The smoothness parameter `ν` of the Matérn kernel.
                       As ν → ∞, it approaches the RBF kernel. Must be > 0.
            normalize_std (bool): If True, normalize output to have unit standard deviation.
            boundary_condition (str): One of "zero-neumann", "periodic", or "none".
            device (torch.device): The PyTorch device to use.
        """
        self.in_channels = in_channels
        self.Ln1 = Ln1
        self.Ln2 = Ln2
        self.device = device
        self.normalize_std = normalize_std
        self.boundary_condition = boundary_condition

        # Validate parameters
        if nu <= 0:
            raise ValueError("Smoothness parameter nu must be positive.")
        if scale <= 0:
            raise ValueError("Length scale must be positive.")

        # Store parameters
        self.scale = scale
        self.nu = nu

        if boundary_condition == "zero-neumann":
            if Ln1 != Ln2:
                raise ValueError("MaternKernelSampler with 'zero-neumann' currently supports square grids only")
            self.Ln = Ln1
            self._setup_zero_neumann()
        elif boundary_condition == "periodic":
            self._setup_periodic()
        elif boundary_condition == "none":
            self._setup_none()
        else:
            raise ValueError("boundary_condition must be one of 'zero-neumann', 'periodic', or 'none'")

    @staticmethod
    def _normalize_to_unit_std(tensor):
        """Normalizes a tensor to have a standard deviation of 1."""
        current_std = tensor.std()
        if current_std > 0:  # Avoid division by zero
            return tensor / current_std
        return tensor

    def _setup_zero_neumann(self):
        """Sets up spectral coefficients for zero-Neumann boundary condition using IDCT."""
        k = torch.arange(self.Ln, device=self.device)
        K1, K2 = torch.meshgrid(k, k, indexing="ij")

        # Calculate parameters directly from scale and nu
        D = 2.0  # spatial dimension
        alpha = self.nu + D / 2.0
        tau = np.sqrt(2 * self.nu) / self.scale

        # Define the (square root of) eigenvalues of the covariance operator
        C = (torch.pi**2) * (torch.square(K1) + torch.square(K2)) + tau**2
        C = torch.pow(C, -alpha / 2.0)

        # Calculate the constant factor from Matérn spectral density
        gamma_ratio = poch(self.nu, D / 2)  # gamma(self.nu + n/2) / gamma(self.nu)
        const_factor = np.sqrt((2**D * np.pi ** (D / 2) * gamma_ratio * (2 * self.nu) ** self.nu) / (self.scale ** (2 * self.nu)))

        C = const_factor * C
        self.coeff = C

    def _setup_periodic(self):
        """Sets up spectral coefficients for periodic boundary using rfft frequencies."""
        freq1 = torch.fft.fftfreq(self.Ln1, 1 / self.Ln1, device=self.device)
        freq2 = torch.fft.rfftfreq(self.Ln2, 1 / self.Ln2, device=self.device)  # Note: rfftfreq
        K1, K2 = torch.meshgrid(freq1, freq2, indexing="ij")

        # Calculate parameters directly from scale and nu
        D = 2.0  # spatial dimension
        alpha = self.nu + D / 2.0
        tau = np.sqrt(2 * self.nu) / self.scale

        # Define the (square root of) eigenvalues of the covariance operator
        C = (2 * torch.pi) ** 2 * (torch.square(K1) + torch.square(K2)) + tau**2
        C = torch.pow(C, -alpha / 2.0)

        # Calculate the constant factor from Matérn spectral density
        gamma_ratio = poch(self.nu, D / 2)  # gamma(self.nu + n/2) / gamma(self.nu)
        const_factor = np.sqrt((2**D * np.pi ** (D / 2) * gamma_ratio * (2 * self.nu) ** self.nu) / (self.scale ** (2 * self.nu)))

        C = const_factor * C
        self.coeff_rfft = C

    def _setup_none(self):
        """Sets up a nested sampler on a larger grid to simulate no boundary condition."""
        # This works by generating a larger periodic field and cropping the center,
        # which mitigates wrap-around effects from the periodic boundary.
        self.nested_sampler = MaternKernelSampler(
            in_channels=self.in_channels,
            Ln1=2 * self.Ln1,
            Ln2=2 * self.Ln2,
            scale=self.scale / 2,  # Corrected scale for larger grid
            nu=self.nu,
            normalize_std=False,  # Normalization is handled after cropping
            boundary_condition="periodic",
            device=self.device,
        )

    @torch.no_grad()
    def sample(self, N):
        """
        Generates N samples of the GRF using the specified boundary condition.

        Returns:
            torch.Tensor: A tensor of shape (N, in_channels, Ln1, Ln2).
        """
        if self.boundary_condition == "zero-neumann":
            return self._sample_zero_neumann(N)
        elif self.boundary_condition == "periodic":
            return self._sample_periodic(N)
        else:  # "none"
            return self._sample_none(N)

    def _sample_zero_neumann(self, N):
        """Samples using IDCT for zero-Neumann boundary."""
        xr = torch.randn(N, self.in_channels, self.Ln, self.Ln, device=self.device)

        # Apply spectral filter (coefficients)
        L = self.coeff[None, None, :, :] * xr
        L = self.Ln * L

        # Transform to real domain using IDCT
        result = idct_2d(L, norm="ortho")

        if self.normalize_std:
            result = self._normalize_to_unit_std(result)
        return result

    def _sample_periodic(self, N):
        """Samples using IFFT for periodic boundary."""
        # Shape: (N, in_channels, Ln1, Ln2 // 2 + 1)
        xr = torch.randn(N, self.in_channels, self.Ln1, self.Ln2 // 2 + 1, 2, device=self.device)
        xr = torch.view_as_complex(xr) / np.sqrt(2)  # standard complex noise

        # Apply spectral filter
        L = self.coeff_rfft[None, None, :, :] * xr

        # Special handling for DC and Nyquist frequencies
        L[:, :, 0, 0] = L[:, :, 0, 0].real * np.sqrt(2)  # DC must be real
        if self.Ln2 % 2 == 0:
            L[:, :, :, -1] = L[:, :, :, -1].real * np.sqrt(2)  # Nyquist must be real
        if self.Ln1 % 2 == 0:
            L[:, :, self.Ln1 // 2, 0] = L[:, :, self.Ln1 // 2, 0].real * np.sqrt(2)
            if self.Ln2 % 2 == 0:
                L[:, :, self.Ln1 // 2, -1] = L[:, :, self.Ln1 // 2, -1].real * np.sqrt(2)

        # Transform to real domain using IRFFT2
        result = torch.fft.irfft2(L, s=(self.Ln1, self.Ln2), dim=(-2, -1), norm="forward")

        if self.normalize_std:
            result = self._normalize_to_unit_std(result)
        return result

    def _sample_none(self, N):
        """Samples by cropping from a larger periodic field."""
        large_samples = self.nested_sampler.sample(N)

        # Crop the central region to get the desired grid size
        start1 = self.Ln1 // 2
        end1 = start1 + self.Ln1
        start2 = self.Ln2 // 2
        end2 = start2 + self.Ln2
        result = large_samples[:, :, start1:end1, start2:end2]

        if self.normalize_std:
            result = self._normalize_to_unit_std(result)
        return result
