from ..normalizers import UnitGaussianNormalizer
import torch
from torch.testing import assert_close
from flaky import flaky

@flaky(max_runs=4, min_passes=3)
def test_UnitGaussianNormalizer_created_from_stats(eps=1e-6):
    x = torch.rand(16, 3, 40, 50, 60)*2.5
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
    x = torch.rand(16, 3, 40, 50, 60)*2.5
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
    x = torch.rand(16, 3, 40, 50, 60)*2.5
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

