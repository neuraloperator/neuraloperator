from ..output_encoder import UnitGaussianNormalizer
import torch
from torch.testing import assert_close

def test_UnitGaussianNormalizer():
    x = torch.randn((2, 3, 4, 5, 6))
    mean = torch.mean(x, dim=[2, 3, 4], keepdim=True)
    std = torch.std(x, dim=[2, 3, 4], keepdim=True)

    normalizer = UnitGaussianNormalizer(mean=mean, std=std)
    x_normalized = normalizer.encode(x)
    x_unnormalized = normalizer.decode(x_normalized)

    eps = 1e-7
    assert_close(x_unnormalized, x)
    assert torch.mean(x_normalized) <= eps
    assert (torch.std(x_normalized) - 1) <= eps

    normalizer = UnitGaussianNormalizer.from_data(x)
    x_normalized = normalizer.encode(x)
    x_unnormalized = normalizer.decode(x_normalized)

    eps = 1e-7
    assert_close(x_unnormalized, x)
    assert torch.mean(x_normalized) <= eps
    assert (torch.std(x_normalized) - 1) <= eps
