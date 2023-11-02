from ..output_encoder import UnitGaussianNormalizer
import torch
from torch.testing import assert_close


def test_UnitGaussianNormalizer():
    x = torch.rand(4, 3, 4, 5, 6)*2.5
    mean = torch.mean(x, dim=[0, 2, 3, 4], keepdim=True)
    std = torch.std(x, dim=[0, 2, 3, 4], keepdim=True)

    # Init normalizer with ground-truth mean and std
    normalizer = UnitGaussianNormalizer(mean=mean, std=std)
    x_normalized = normalizer.transform(x)
    x_unnormalized = normalizer.inverse_transform(x_normalized)

    eps = 1e-5
    assert_close(x_unnormalized, x)
    assert torch.mean(x_normalized) <= eps
    assert (torch.std(x_normalized) - 1) <= eps

    # Init by fitting whole data at once
    normalizer = UnitGaussianNormalizer(dim=[0, 2, 3, 4])
    normalizer.fit(x)
    x_normalized = normalizer.transform(x)
    x_unnormalized = normalizer.inverse_transform(x_normalized)

    eps = 1e-3
    assert_close(x_unnormalized, x)
    assert torch.mean(x_normalized) <= eps
    assert (torch.std(x_normalized) - 1) <= eps

    assert_close(normalizer.mean, mean)
    assert_close(normalizer.std, std, rtol=1e-3, atol=1e-3)

    # Incrementally compute mean and var
    normalizer = UnitGaussianNormalizer(dim=[0, 2, 3, 4])
    normalizer.partial_fit(x, batch_size=2)
    x_normalized = normalizer.transform(x)
    x_unnormalized = normalizer.inverse_transform(x_normalized)

    eps = 1e-3
    assert_close(x_unnormalized, x)
    assert torch.mean(x_normalized) <= eps
    assert (torch.std(x_normalized) - 1) <= eps

    assert_close(normalizer.mean, mean)
    print(normalizer.std, std)
    assert_close(normalizer.std, std, rtol=1e-2, atol=1e-2)

