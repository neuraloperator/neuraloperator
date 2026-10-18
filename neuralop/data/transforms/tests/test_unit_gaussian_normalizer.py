from ..normalizers import UnitGaussianNormalizer
import torch
import numpy as np
import pytest
from torch.testing import assert_close
from flaky import flaky


@flaky(max_runs=4, min_passes=3)
def test_UnitGaussianNormalizer_created_from_stats(eps=1e-6):
    x = torch.rand(16, 3, 40, 50, 60) * 2.5
    mean = torch.mean(x, dim=[0, 2, 3, 4], keepdim=True)
    std = torch.std(x, dim=[0, 2, 3, 4], keepdim=True)

    # Init normalizer with ground-truth mean and std
    normalizer = UnitGaussianNormalizer(mean=mean, std=std, eps=eps)
    x_normalized = normalizer.transform(x)
    x_unnormalized = normalizer.inverse_transform(x_normalized)

    assert_close(x_unnormalized, x)
    assert torch.mean(x_normalized) <= eps
    assert (torch.std(x_normalized) - 1) <= eps


@flaky(max_runs=4, min_passes=3)
def test_UnitGaussianNormalizer_from_data(eps=1e-6):
    x = torch.rand(16, 3, 40, 50, 60) * 2.5
    mean = torch.mean(x, dim=[0, 2, 3, 4], keepdim=True)
    std = torch.std(x, dim=[0, 2, 3, 4], keepdim=True)
    # Init by fitting whole data at once
    normalizer = UnitGaussianNormalizer(dim=[0, 2, 3, 4], eps=eps)
    normalizer.fit(x)

    assert_close(normalizer.mean, mean)
    assert_close(normalizer.std, std, rtol=eps, atol=eps)

    x_normalized = normalizer.transform(x)
    x_unnormalized = normalizer.inverse_transform(x_normalized)

    assert_close(x_unnormalized, x)
    assert torch.mean(x_normalized) <= eps
    assert (torch.std(x_normalized) - 1) <= eps

    assert_close(normalizer.mean, mean)
    assert_close(normalizer.std, std, rtol=eps, atol=eps)


@flaky(max_runs=4, min_passes=3)
def test_UnitGaussianNormalizer_incremental_update(eps=1e-6):
    x = torch.rand(16, 3, 40, 50, 60) * 2.5
    mean = torch.mean(x, dim=[0, 2, 3, 4], keepdim=True)
    std = torch.std(x, dim=[0, 2, 3, 4], keepdim=True)
    # Incrementally compute mean and var
    normalizer = UnitGaussianNormalizer(dim=[0, 2, 3, 4], eps=eps)
    normalizer.partial_fit(x, batch_size=2)

    x_normalized = normalizer.transform(x)
    x_unnormalized = normalizer.inverse_transform(x_normalized)

    assert_close(x_unnormalized, x)
    assert torch.mean(x_normalized) <= eps
    assert (torch.std(x_normalized) - 1) <= eps

    assert_close(normalizer.mean, mean)
    assert_close(normalizer.std, std, rtol=eps, atol=eps)


def _assert_sample_stats(normalizer, data, dim):
    """Compare streaming statistics with independent sample statistics."""
    axis = tuple(dim) if isinstance(dim, list) else dim
    expected_mean = np.mean(data.numpy(), axis=axis, keepdims=True)
    expected_std = np.std(data.numpy(), axis=axis, ddof=1, keepdims=True)
    assert_close(
        normalizer.mean, torch.from_numpy(expected_mean), rtol=1e-12, atol=1e-12
    )
    assert_close(normalizer.std, torch.from_numpy(expected_std), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dim", [None, 0, [0, 2]])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
def test_incremental_sample_std_matches_full_data(dim, batch_size):
    """Batch partitioning must preserve the sample standard deviation."""
    data = torch.arange(30, dtype=torch.float64).reshape(5, 2, 3)
    normalizer = UnitGaussianNormalizer(dim=dim)
    normalizer.partial_fit(data, batch_size=batch_size)
    _assert_sample_stats(normalizer, data, dim)


@pytest.mark.parametrize("dim", [None, 0, [0, 2]])
def test_incremental_sample_std_after_fit(dim):
    """Extending whole-batch statistics uses the same sample correction."""
    data = torch.arange(30, dtype=torch.float64).reshape(5, 2, 3)
    normalizer = UnitGaussianNormalizer(dim=dim)
    normalizer.fit(data[:2])
    normalizer.partial_fit(data[2:], batch_size=2)
    _assert_sample_stats(normalizer, data, dim)


@pytest.mark.parametrize("dim", [None, 0, [0, 2]])
def test_dataset_sample_std_matches_full_data(dim):
    """Dataset-created normalizers use consistent sample statistics."""
    data = torch.arange(30, dtype=torch.float64).reshape(5, 2, 3)
    dataset = [{"field": sample} for sample in data]
    normalizer = UnitGaussianNormalizer.from_dataset(dataset, dim=dim, keys=["field"])[
        "field"
    ]
    _assert_sample_stats(normalizer, data, dim)
