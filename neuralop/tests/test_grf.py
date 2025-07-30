import torch
import numpy as np
import pytest
from torch.testing import assert_close

from neuralop.grf import RBFKernelSampler, MaternKernelSampler


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
        sampler = RBFKernelSampler(
            in_channels=1,
            Ln1=64,
            Ln2=64,
            scale=0.1,
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
            scale=0.1,
            nu=2.5,
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
                    scale=0.1,
                    nu=2.5,
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
            scale=0.1,
            nu=2.5,
            normalize_std=True,
            boundary_condition=boundary_condition,
            device=device,
        )

        samples_norm = sampler_norm.sample(100)
        std_norm = samples_norm.std().item()

        # Should be approximately 1 due to normalization
        assert abs(std_norm - 1.0) < 0.01

    @pytest.mark.parametrize("boundary_condition", ["zero-neumann", "periodic", "none"])
    def test_matern_sampler_reproducibility(self, boundary_condition):
        """Test that sampler produces consistent results with same random seed"""

        sampler = MaternKernelSampler(
            in_channels=1,
            Ln1=64,
            Ln2=64,
            scale=0.1,
            nu=2.5,
            boundary_condition=boundary_condition,
        )

        # Set seed and sample
        np.random.seed(42)
        samples1 = sampler.sample(2)

        # Reset seed and sample again
        np.random.seed(42)
        samples2 = sampler.sample(2)

        assert_close(samples1, samples2)


if __name__ == "__main__":
    pytest.main([__file__])
