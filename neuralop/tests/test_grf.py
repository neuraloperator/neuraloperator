import torch
import numpy as np
import pytest
from torch.testing import assert_close

from ..grf import RBFKernelSampler, MaternKernelSampler, get_fixed_coords


class TestGetFixedCoords:
    def test_get_fixed_coords_shape(self):
        """Test that get_fixed_coords returns correct shape"""
        Ln1, Ln2 = 32, 64
        coords = get_fixed_coords(Ln1, Ln2)
        expected_shape = (Ln1 * Ln2, 2)
        assert coords.shape == expected_shape

    def test_get_fixed_coords_range(self):
        """Test that coordinates are in [0, 1) range"""
        Ln1, Ln2 = 16, 16
        coords = get_fixed_coords(Ln1, Ln2)
        assert coords.min() >= 0.0
        assert coords.max() < 1.0

    def test_get_fixed_coords_square_grid(self):
        """Test specific values for a small square grid"""
        Ln1, Ln2 = 2, 2
        coords = get_fixed_coords(Ln1, Ln2)
        expected = torch.tensor([[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]])
        assert_close(coords, expected)


class TestRBFKernelSampler:
    @pytest.mark.parametrize("device", [torch.device("cpu")])
    @pytest.mark.parametrize("in_channels", [1, 2])
    @pytest.mark.parametrize("grid_size", [16, 32])
    def test_rbf_sampler_output_shape(self, device, in_channels, grid_size):
        """Test RBF sampler output shape"""
        sampler = RBFKernelSampler(
            in_channels=in_channels,
            Ln1=grid_size,
            Ln2=grid_size,
            scale=0.1,
            device=device,
        )

        N = 4
        samples = sampler.sample(N)
        expected_shape = (N, in_channels, grid_size, grid_size)
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
        N = 100
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
    @pytest.mark.parametrize(
        "in_channels",
        [
            1,
            2,
        ],
    )
    @pytest.mark.parametrize("grid_size", [16, 32])
    def test_matern_sampler_output_shape(self, device, in_channels, grid_size):
        """Test Matern sampler output shape"""
        sampler = MaternKernelSampler(
            in_channels=in_channels,
            Ln1=grid_size,
            Ln2=grid_size,
            alpha=2,
            tau=3,
            device=device,
        )

        N = 4
        samples = sampler.sample(N)
        expected_shape = (N, in_channels, grid_size, grid_size)
        assert samples.shape == expected_shape
        assert samples.device == device

    def test_matern_sampler_normalization(self):
        """Test normalization functionality"""
        device = torch.device("cpu")

        # Test with normalization
        sampler_norm = MaternKernelSampler(
            in_channels=1, Ln1=32, Ln2=32, normalize_std=True, device=device
        )

        samples_norm = sampler_norm.sample(10)
        std_norm = samples_norm.std().item()

        # Should be approximately 1 due to normalization
        assert abs(std_norm - 1.0) < 0.01

    def test_matern_sampler_reproducibility(self):
        """Test that sampler produces consistent results with same random seed"""
        device = torch.device("cpu")
        sampler = MaternKernelSampler(in_channels=1, Ln1=16, Ln2=16, device=device)

        # Set seed and sample
        np.random.seed(42)
        samples1 = sampler.sample(2)

        # Reset seed and sample again
        np.random.seed(42)
        samples2 = sampler.sample(2)

        assert_close(samples1, samples2)


if __name__ == "__main__":
    pytest.main([__file__])
