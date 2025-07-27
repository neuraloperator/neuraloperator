import torch
import numpy as np
import pytest
from torch.testing import assert_close

from ..grf import RBFKernelSampler, MaternKernelSampler


class TestRBFKernelSampler:
    @pytest.mark.parametrize("device", [torch.device("cpu")])
    @pytest.mark.parametrize("in_channels", [1, 2])
    @pytest.mark.parametrize("grid_size", [(32, 32), (64, 60)])
    def test_rbf_sampler_output_shape(self, device, in_channels, grid_size):
        """Test RBF sampler output shape"""
        Ln1, Ln2 = grid_size
        sampler = RBFKernelSampler(
            in_channels=in_channels,
            Ln1=Ln1,
            Ln2=Ln2,
            scale=0.1,
            device=device,
        )

        N = 4
        samples = sampler.sample(N)
        expected_shape = (N, in_channels, Ln1, Ln2)
        assert samples.shape == expected_shape
        assert samples.device == device

    def test_rbf_sampler_normalization(self):
        """Test that RBF sampler produces samples with zero mean and unit std"""
        device = torch.device("cpu")
        sampler = RBFKernelSampler(
            in_channels=1,
            Ln1=64,
            Ln2=64,
            scale=0.1,
            device=device,
        )

        # Generate enough samples for reliable statistics
        N = 500
        samples = sampler.sample(N)

        # Calculate statistics
        mean = samples.mean().item()
        std = samples.std().item()

        # Mean should be approximately zero (within tolerance due to randomness)
        assert abs(mean) < 0.01, f"Expected mean ≈ 0, got {mean}"

        # Standard deviation should be approximately 1
        assert abs(std - 1.0) < 0.01, f"Expected std ≈ 1, got {std}"


class TestMaternKernelSampler:
    @pytest.mark.parametrize("device", [torch.device("cpu")])
    @pytest.mark.parametrize("in_channels", [1, 2])
    def test_matern_sampler_output_shape(self, device, in_channels):
        """Test Matern sampler output shape with appropriate grid sizes for each boundary condition"""

        # Test zero-neumann with square grid only
        sampler_zn = MaternKernelSampler(
            in_channels=in_channels,
            Ln1=64,
            Ln2=64,
            alpha=2,
            tau=3,
            boundary_condition="zero-neumann",
            device=device,
        )

        N = 4
        samples_zn = sampler_zn.sample(N)
        expected_shape_zn = (N, in_channels, 64, 64)
        assert samples_zn.shape == expected_shape_zn
        assert samples_zn.device == device

        # Test periodic and none with various grid sizes
        for boundary_condition in ["periodic", "none"]:
            for grid_size in [(32, 32), (64, 60)]:
                Ln1, Ln2 = grid_size
                sampler = MaternKernelSampler(
                    in_channels=in_channels,
                    Ln1=Ln1,
                    Ln2=Ln2,
                    alpha=2,
                    tau=3,
                    boundary_condition=boundary_condition,
                    device=device,
                )

                samples = sampler.sample(N)
                expected_shape = (N, in_channels, Ln1, Ln2)
                assert samples.shape == expected_shape
                assert samples.device == device

    @pytest.mark.parametrize("boundary_condition", ["zero-neumann", "periodic", "none"])
    def test_matern_sampler_normalization(self, boundary_condition):
        """Test normalization functionality"""
        device = torch.device("cpu")

        # Test with normalization
        sampler_norm = MaternKernelSampler(
            in_channels=1,
            Ln1=64,
            Ln2=64,
            normalize_std=True,
            boundary_condition=boundary_condition,
            device=device,
        )

        samples_norm = sampler_norm.sample(10)
        std_norm = samples_norm.std().item()

        # Should be approximately 1 due to normalization
        assert abs(std_norm - 1.0) < 0.01

    @pytest.mark.parametrize("boundary_condition", ["zero-neumann", "periodic", "none"])
    def test_matern_sampler_reproducibility(self, boundary_condition):
        """Test that sampler produces consistent results with same random seed"""
        device = torch.device("cpu")

        sampler = MaternKernelSampler(
            in_channels=1,
            Ln1=64,
            Ln2=64,
            boundary_condition=boundary_condition,
            device=device,
        )

        # Set seed and sample
        np.random.seed(42)
        samples1 = sampler.sample(2)

        # Reset seed and sample again
        np.random.seed(42)
        samples2 = sampler.sample(2)

        assert_close(samples1, samples2)

    def _calculate_theoretical_spectrum(self, Ln1, Ln2, alpha, tau):
        """Calculates the theoretical Matérn power spectrum from first principles."""
        # Create frequency grids
        freq1 = np.fft.fftfreq(Ln1, 1 / Ln1)
        freq2 = np.fft.fftfreq(Ln2, 1 / Ln2)
        K1, K2 = np.meshgrid(freq1, freq2, indexing="ij")

        # Squared wavenumbers (kappa^2)
        kappa_sq = (2 * np.pi) ** 2 * (np.square(K1) + np.square(K2))

        # Power spectrum corresponds to the eigenvalues of the covariance operator C
        # C = (-Delta + tau^2)^(-alpha)
        # Eigenvalues are (kappa^2 + tau^2)^(-alpha)
        spectrum = np.power(kappa_sq + tau**2, -alpha)
        spectrum *= tau ** (2 * (alpha - 1))

        # Set DC component to zero, as is convention for these GRFs
        spectrum[0, 0] = 0.0

        return torch.from_numpy(spectrum)

    def _calculate_radial_average(self, spectrum):
        """Calculate radial average of a 2D power spectrum."""
        Ln1, Ln2 = spectrum.shape
        center_x, center_y = Ln1 // 2, Ln2 // 2

        # Create coordinate grids
        y, x = torch.meshgrid(torch.arange(Ln1), torch.arange(Ln2), indexing="ij")

        # Calculate distance from center
        distances = torch.sqrt(
            (x - center_y).float() ** 2 + (y - center_x).float() ** 2
        )

        # Define radial bins
        max_radius = min(center_x, center_y)
        radial_bins = torch.arange(0, max_radius, dtype=torch.float32)

        # Calculate radial average
        radial_avg = torch.zeros_like(radial_bins)
        for i, r in enumerate(radial_bins):
            if i == len(radial_bins) - 1:
                mask = distances >= r
            else:
                mask = (distances >= r) & (distances < radial_bins[i + 1])

            if mask.sum() > 0:
                radial_avg[i] = spectrum[mask].mean()

        return radial_bins, radial_avg

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    @pytest.mark.parametrize("boundary_condition", ["periodic", "none"])
    @pytest.mark.parametrize("alpha, tau", [(2.0, 3.0), (3.0, 5.0)])
    def test_matern_sampler_spectrum(self, device, boundary_condition, alpha, tau):
        """
        Test that the radial average of the statistical energy spectrum of generated
        samples matches the theoretical spectrum defined by alpha and tau.
        This test treats the sampler as a black box.
        """
        Ln1, Ln2 = 64, 64
        in_channels = 1

        sampler = MaternKernelSampler(
            in_channels=in_channels,
            Ln1=Ln1,
            Ln2=Ln2,
            alpha=alpha,
            tau=tau,
            boundary_condition=boundary_condition,
            device=device,
            normalize_std=False,  # Normalization rescales the spectrum, so disable it
        )

        # --- Calculate Theoretical Spectrum ---
        # We calculate the ideal spectrum for the final output grid size.
        theoretical_power_spectrum = self._calculate_theoretical_spectrum(
            Ln1, Ln2, alpha, tau
        ).to(device)

        # Generate a large number of samples for reliable statistical analysis
        N = 500  # Increased N for better statistics
        samples = sampler.sample(N)

        # --- Calculate Empirical Spectrum ---
        # We use the 2D Fast Fourier Transform (FFT) on the final output samples.
        fft_coeffs = torch.fft.fft2(samples, norm="ortho")

        # The empirical power spectrum is the mean of the squared magnitude of the FFT coefficients.
        empirical_power_spectrum = torch.mean(
            torch.abs(fft_coeffs).pow(2), dim=0
        ).squeeze(0)

        # --- Calculate Radial Averages ---
        radial_bins_theoretical, radial_avg_theoretical = (
            self._calculate_radial_average(theoretical_power_spectrum.float())
        )
        radial_bins_empirical, radial_avg_empirical = self._calculate_radial_average(
            empirical_power_spectrum
        )

        # Compare the radial averages. This is more robust than 2D comparison
        # as it reduces statistical noise and focuses on the essential radial decay.
        # Skip the first bin (r=0) as it corresponds to the DC component
        assert_close(
            radial_avg_empirical[1:],
            radial_avg_theoretical[1:],
            rtol=0.1,
            atol=1e-4,
        )


if __name__ == "__main__":
    pytest.main([__file__])
