import torch
import numpy as np
from scipy.fft import idctn, ifftn


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
        samples = torch.zeros((N, self.Ln1 * self.Ln2, self.in_channels)).to(
            self.device
        )
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
        length_scale=0.1,
        nu=1.0,
        normalize_std=True,
        boundary_condition="zero-neumann",
        device=None,
    ):
        """
        Gaussian random field sampler for a Matérn covariance function, implemented
        using spectral methods.

        The Matérn covariance is defined by a length scale `l` and a smoothness
        parameter `ν`. The covariance operator is C = (-Δ + τ²)^(-α), where Δ is
        the Laplacian. The parameters are related via:
        - α = ν + D/2  (D=2 for a 2D grid)
        - τ² = 2ν / l²

        Args:
            in_channels (int): Number of input channels for the samples.
            Ln1, Ln2 (int): Grid dimensions.
            length_scale (float): The length scale `l` of the Matérn kernel.
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
        self.length_scale = length_scale
        self.nu = nu
        self.normalize_std = normalize_std
        self.boundary_condition = boundary_condition

        if self.nu <= 0:
            raise ValueError("Smoothness parameter nu must be positive.")
        if self.length_scale <= 0:
            raise ValueError("Length scale must be positive.")

        # The dimension D of the grid is 2
        D = 2.0
        # Relate (length_scale, nu) to the (alpha, tau) used in the spectral method
        self.alpha = self.nu + D / 2.0
        self.tau = np.sqrt(2 * self.nu) / self.length_scale

        if boundary_condition not in ["zero-neumann", "periodic", "none"]:
            raise ValueError(
                "boundary_condition must be one of 'zero-neumann', 'periodic', or 'none'"
            )

        if boundary_condition == "zero-neumann":
            if Ln1 != Ln2:
                raise ValueError(
                    "MaternKernelSampler with 'zero-neumann' currently supports square grids only"
                )
            self.Ln = Ln1
            self._setup_zero_neumann()
        elif boundary_condition == "periodic":
            self._setup_periodic()
        else:  # "none"
            self._setup_none()

    @staticmethod
    def _normalize_to_unit_std(tensor):
        """Normalizes a tensor to have a standard deviation of 1."""
        current_std = tensor.std()
        if current_std > 0:  # Avoid division by zero
            return tensor / current_std
        return tensor

    @staticmethod
    def _ensure_hermitian_symmetry(L, Ln1, Ln2):
        """Enforces Hermitian symmetry on the spectral coefficients for real-valued output."""
        # For real-valued output from IFFT, we need L[k] = conj(L[-k]).
        # This is explicitly enforced for numerical stability.
        L[:, :, 0, 0] = np.real(L[:, :, 0, 0])  # DC component
        if Ln1 % 2 == 0:
            L[:, :, Ln1 // 2, :] = np.real(L[:, :, Ln1 // 2, :])  # Nyquist frequency
        if Ln2 % 2 == 0:
            L[:, :, :, Ln2 // 2] = np.real(L[:, :, :, Ln2 // 2])  # Nyquist frequency
        return L

    def _setup_zero_neumann(self):
        """Sets up spectral coefficients for zero-Neumann boundary condition using IDCT."""
        k = np.arange(self.Ln)
        K1, K2 = np.meshgrid(k, k, indexing="ij")

        # Define the (square root of) eigenvalues of the covariance operator
        C = (np.pi**2) * (np.square(K1) + np.square(K2)) + self.tau**2
        C = np.power(C, -self.alpha / 2.0)
        C = (self.tau ** (self.alpha - 1)) * C

        self.coeff = C

    def _setup_periodic(self):
        """Sets up spectral coefficients for periodic boundary condition using IFFT."""
        freq1 = np.fft.fftfreq(self.Ln1, 1 / self.Ln1)
        freq2 = np.fft.fftfreq(self.Ln2, 1 / self.Ln2)
        K1, K2 = np.meshgrid(freq1, freq2, indexing="ij")

        # Define the (square root of) eigenvalues for periodic case
        # Using 2π scaling for proper frequency scaling
        C = (2 * np.pi) ** 2 * (np.square(K1) + np.square(K2)) + self.tau**2
        C = np.power(C, -self.alpha / 2.0)
        C = (self.tau ** (self.alpha - 1)) * C

        C[0, 0] = 0.0  # Set DC component to zero
        self.coeff = C

    def _setup_none(self):
        """Sets up a nested sampler on a larger grid to simulate no boundary condition."""
        # This works by generating a larger periodic field and cropping the center,
        # which mitigates wrap-around effects from the periodic boundary.
        self.nested_sampler = MaternKernelSampler(
            in_channels=self.in_channels,
            Ln1=2 * self.Ln1,
            Ln2=2 * self.Ln2,
            length_scale=self.length_scale,
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
        xr = np.random.standard_normal(size=(N, self.in_channels, self.Ln, self.Ln))

        # Apply spectral filter (coefficients)
        L = self.coeff[None, None, :, :] * xr
        L = self.Ln * L

        # Apply boundary condition to all samples
        L[:, :, 0, 0] = 0.0

        # Transform to real domain using IDCT
        u = idctn(L, axes=[-1, -2], norm="ortho")
        result = torch.from_numpy(u.astype(np.float32)).to(self.device)

        if self.normalize_std:
            result = self._normalize_to_unit_std(result)
        return result

    def _sample_periodic(self, N):
        """Samples using IFFT for periodic boundary."""
        xr = np.random.standard_normal(
            size=(N, self.in_channels, self.Ln1, self.Ln2)
        ) + 1j * np.random.standard_normal(
            size=(N, self.in_channels, self.Ln1, self.Ln2)
        )

        # Apply spectral filter (coefficients)
        L = self.coeff[None, None, :, :] * xr
        L = self._ensure_hermitian_symmetry(L, self.Ln1, self.Ln2)

        # Transform to real domain using IFFT
        u = ifftn(L, axes=[-1, -2])
        u = np.real(u) * np.sqrt(self.Ln1 * self.Ln2)  # Take real part and scale
        result = torch.from_numpy(u.astype(np.float32)).to(self.device)

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
