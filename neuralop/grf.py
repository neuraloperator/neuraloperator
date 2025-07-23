import torch
import numpy as np
import cv2


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
        alpha=2,
        tau=3,
        normalize_std=True,
        device=None,
    ):
        """
        Gaussian random field Non-Periodic Boundary using IDCT method.
        covariance operator C = (-Delta + tau^2)^(-alpha)
        Delta is the Laplacian with zero Neumann boundary condition

        Args:
            in_channels: Number of input channels
            Ln1, Ln2: Grid dimensions
            alpha: Smoothness parameter (higher = smoother)
            tau: Length scale parameter
            normalize_std: If True, normalize output to have unit standard deviation
            device: PyTorch device
        """
        self.in_channels = in_channels
        assert Ln1 == Ln2, "MaternKernelSampler currently supports square grids only"
        self.Ln = Ln1
        self.device = device
        self.alpha = alpha
        self.tau = tau
        self.normalize_std = normalize_std

        # Create wavenumber grid
        k = np.arange(self.Ln)
        K1, K2 = np.meshgrid(k, k)

        # Define the (square root of) eigenvalues of the covariance operator
        C = (np.pi**2) * (np.square(K1) + np.square(K2)) + tau**2
        C = np.power(C, -alpha / 2.0)
        C = (tau ** (alpha - 1)) * C

        # Store coefficient
        self.coeff = C

    def _sample2d(self):
        """
        Single 2D Sample
        :return: GRF numpy.ndarray (Ln, Ln)
        """
        # Sample from normal distribution
        xr = np.random.standard_normal(size=(self.Ln, self.Ln))
        # Coefficients in fourier domain
        L = self.coeff * xr
        L = self.Ln * L
        # Apply boundary condition
        L[0, 0] = 0.0
        # Transform to real domain
        u = cv2.idct(L)
        return u

    @torch.no_grad()
    def sample(self, N):
        """
        Generate N samples of GRF using IDCT method.

        Returns:
            Tensor of shape (N, in_channels, Ln1, Ln2)
        """
        # Generate samples for all channels
        z_mat = np.zeros((N, self.in_channels, self.Ln, self.Ln), dtype=np.float32)

        for n in range(N):
            for c in range(self.in_channels):
                z_mat[n, c, :, :] = self._sample2d()

        # Convert to torch tensor
        result = torch.from_numpy(z_mat)
        if self.device is not None:
            result = result.to(self.device)

        # Normalize to unit standard deviation if requested
        if self.normalize_std:
            current_std = result.std()
            if current_std > 0:  # Avoid division by zero
                result = result / current_std

        return result
