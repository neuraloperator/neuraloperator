import torch
import numpy as np
from scipy.fft import idctn, ifftn


def get_fixed_coords(Ln1, Ln2):
    xs = torch.linspace(0, 1, steps=Ln1 + 1)[0:-1]
    ys = torch.linspace(0, 1, steps=Ln2 + 1)[0:-1]
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    coords = torch.cat([yy.reshape(-1, 1), xx.reshape(-1, 1)], dim=-1)
    return coords


class GRFSampler(object):
    def sample(self, N):
        raise NotImplementedError()


class RBFKernelSampler(GRFSampler):
    @torch.no_grad()
    def __init__(self, in_channels, Ln1, Ln2, scale=1, eps=0.01, device=None):
        self.in_channels = in_channels
        self.Ln1 = Ln1
        self.Ln2 = Ln2
        self.device = device
        self.scale = scale

        # (s^2, 2)
        meshgrid = get_fixed_coords(self.Ln1, self.Ln2).to(device)
        # (s^2, s^2)
        C = torch.exp(-torch.cdist(meshgrid, meshgrid) / (2 * scale**2))
        # Need to add some regularisation or else the sqrt won't exist
        I = torch.eye(C.size(-1)).to(device)

        # Not memory efficient
        # C = C + (eps**2) * I
        I.mul_(eps**2)  # inplace multiply by eps**2
        C.add_(I)  # inplace add by I
        del I  # don't need it anymore

        self.L = torch.linalg.cholesky(C)
        del C  # save memory

    @torch.no_grad()
    def sample(self, N):
        # (N, s^2, s^2) x (N, s^2, 1) -> (N, s^2, 2)
        # We can do this in one big torch.bmm, but I am concerned about memory
        # so let's just do it iteratively.
        # L_padded = self.L.repeat(N, 1, 1)
        # z_mat = torch.randn((N, self.Ln1*self.Ln2, 2)).to(self.device)
        # sample = torch.bmm(L_padded, z_mat)
        samples = torch.zeros((N, self.Ln1 * self.Ln2, self.in_channels)).to(
            self.device
        )
        for ix in range(N):
            # (s^2, s^2) * (s^2, 2) -> (s^2, 2)
            this_z = torch.randn(self.Ln1 * self.Ln2, self.in_channels).to(self.device)
            samples[ix] = torch.matmul(self.L, this_z)

        # reshape into (N, s, s, n_in)
        sample_rshp = samples.reshape(-1, self.Ln1, self.Ln2, self.in_channels)

        # reshape into (N, n_in, s, s)
        sample_rshp = sample_rshp.transpose(-1, -2).transpose(-2, -3)

        return sample_rshp


class MaternKernelSampler(GRFSampler):
    @torch.no_grad()
    def __init__(
        self,
        in_channels,
        Ln1,
        Ln2,
        alpha=1,
        tau=1,
        normalize_std=True,
        boundary_condition="zero-neumann",
        device=None,
    ):
        """
        Gaussian random field sampler using different boundary conditions.

        covariance operator C = (-Delta + tau^2)^(-alpha)

        Args:
            in_channels: Number of input channels
            Ln1, Ln2: Grid dimensions
            alpha: Smoothness parameter (higher = smoother)
            tau: Length scale parameter
            normalize_std: If True, normalize output to have unit standard deviation
            boundary_condition: Either "zero-neumann", "periodic", or "none"
            device: PyTorch device
        """
        self.in_channels = in_channels
        self.Ln1 = Ln1
        self.Ln2 = Ln2
        self.device = device
        self.alpha = alpha
        self.tau = tau
        self.normalize_std = normalize_std
        self.boundary_condition = boundary_condition

        if boundary_condition not in ["zero-neumann", "periodic", "none"]:
            raise ValueError(
                "boundary_condition must be either 'zero-neumann', 'periodic', or 'none'"
            )

        if boundary_condition == "zero-neumann":
            if Ln1 != Ln2:
                raise ValueError(
                    "MaternKernelSampler with zero-neumann boundary currently supports square grids only"
                )
            self.Ln = Ln1
            self._setup_zero_neumann()
        elif boundary_condition == "periodic":
            self._setup_periodic()
        else:  # none
            self._setup_none()

    @staticmethod
    def _normalize_to_unit_std(tensor):
        current_std = tensor.std()
        if current_std > 0:  # Avoid division by zero
            return tensor / current_std
        return tensor

    @staticmethod
    def _ensure_hermitian_symmetry(L, Ln1, Ln2):
        """Ensure Hermitian symmetry for real-valued output."""
        # For real-valued output, we need L[k] = conj(L[-k])
        # This is automatically satisfied for random complex numbers in most cases,
        # but we explicitly enforce it here for numerical stability

        # Handle DC component (should be real)
        L[:, :, 0, 0] = np.real(L[:, :, 0, 0])

        # Handle Nyquist frequencies if grid size is even
        if Ln1 % 2 == 0:
            L[:, :, Ln1 // 2, :] = np.real(L[:, :, Ln1 // 2, :])
        if Ln2 % 2 == 0:
            L[:, :, :, Ln2 // 2] = np.real(L[:, :, :, Ln2 // 2])

        return L

    def _setup_zero_neumann(self):
        """Setup coefficients for zero Neumann boundary condition using IDCT."""
        # Create wavenumber grid
        k = np.arange(self.Ln)
        K1, K2 = np.meshgrid(k, k)

        # Define the (square root of) eigenvalues of the covariance operator
        C = (np.pi**2) * (np.square(K1) + np.square(K2)) + self.tau**2
        C = np.power(C, -self.alpha / 2.0)
        C = (self.tau ** (self.alpha - 1)) * C

        # Store coefficient
        self.coeff = C

    def _setup_periodic(self):
        """Setup coefficients for periodic boundary condition using FFT."""
        # Create frequency grids for periodic boundary
        freq1 = np.fft.fftfreq(self.Ln1, 1 / self.Ln1)
        freq2 = np.fft.fftfreq(self.Ln2, 1 / self.Ln2)
        K1, K2 = np.meshgrid(freq1, freq2, indexing="ij")

        # Define the (square root of) eigenvalues for periodic case
        # Using 2π scaling for proper frequency scaling
        C = (2 * np.pi) ** 2 * (np.square(K1) + np.square(K2)) + self.tau**2
        C = np.power(C, -self.alpha / 2.0)
        C = (self.tau ** (self.alpha - 1)) * C

        # Handle the zero frequency mode
        C[0, 0] = 0.0

        # Store coefficient
        self.coeff = C

    def _setup_none(self):
        """Setup for 'none' boundary condition using nested sampler with larger grid."""
        # Create a nested MaternKernelSampler with periodic boundary condition
        # and double the grid size
        self.nested_sampler = MaternKernelSampler(
            in_channels=self.in_channels,
            Ln1=2 * self.Ln1,
            Ln2=2 * self.Ln2,
            alpha=self.alpha,
            tau=self.tau,
            normalize_std=False,  # We'll handle normalization at the end
            boundary_condition="periodic",
            device=self.device,
        )

    @torch.no_grad()
    def sample(self, N):
        """
        Generate N samples of GRF using the specified boundary condition.

        Returns:
            Tensor of shape (N, in_channels, Ln1, Ln2)
        """
        if self.boundary_condition == "zero-neumann":
            return self._sample_zero_neumann(N)
        elif self.boundary_condition == "periodic":
            return self._sample_periodic(N)
        else:  # none
            return self._sample_none(N)

    def _sample_zero_neumann(self, N):
        """Sample using IDCT for zero Neumann boundary condition."""
        # Generate all samples at once: (N, in_channels, Ln, Ln)
        xr = np.random.standard_normal(size=(N, self.in_channels, self.Ln, self.Ln))

        # Apply coefficients in fourier domain (broadcasting across N and in_channels)
        L = self.coeff[None, None, :, :] * xr
        L = self.Ln * L

        # Apply boundary condition to all samples
        L[:, :, 0, 0] = 0.0

        # Transform to real domain using vectorized IDCT
        # Apply IDCT along the last two dimensions
        u = idctn(L, axes=[-1, -2], norm="ortho")

        # Convert to torch tensor
        result = torch.from_numpy(u.astype(np.float32))
        if self.device is not None:
            result = result.to(self.device)

        # Normalize to unit standard deviation if requested
        if self.normalize_std:
            result = self._normalize_to_unit_std(result)

        return result

    def _sample_periodic(self, N):
        """Sample using FFT for periodic boundary condition."""
        # Generate complex noise for periodic case
        # Real and imaginary parts
        xr_real = np.random.standard_normal(
            size=(N, self.in_channels, self.Ln1, self.Ln2)
        )
        xr_imag = np.random.standard_normal(
            size=(N, self.in_channels, self.Ln1, self.Ln2)
        )

        # Create complex noise
        xr = xr_real + 1j * xr_imag

        # Apply coefficients in fourier domain
        L = self.coeff[None, None, :, :] * xr

        # Ensure Hermitian symmetry for real output
        # This is important for getting real-valued samples
        L = self._ensure_hermitian_symmetry(L, self.Ln1, self.Ln2)

        # Transform to real domain using IFFT
        u = ifftn(L, axes=[-1, -2])

        # Take real part and scale
        u = np.real(u) * np.sqrt(self.Ln1 * self.Ln2)

        # Convert to torch tensor
        result = torch.from_numpy(u.astype(np.float32))
        if self.device is not None:
            result = result.to(self.device)

        # Normalize to unit standard deviation if requested
        if self.normalize_std:
            result = self._normalize_to_unit_std(result)

        return result

    def _sample_none(self, N):
        """Sample using nested sampler with larger grid and crop central region."""
        # Use the nested sampler to generate samples on the larger grid
        large_samples = self.nested_sampler.sample(N)

        # Crop to central N×N region
        Ln1_large = 2 * self.Ln1
        Ln2_large = 2 * self.Ln2

        start1 = Ln1_large // 4
        end1 = start1 + self.Ln1
        start2 = Ln2_large // 4
        end2 = start2 + self.Ln2

        result = large_samples[:, :, start1:end1, start2:end2]

        # Normalize to unit standard deviation if requested
        if self.normalize_std:
            result = self._normalize_to_unit_std(result)

        return result
