import pytest
import torch


pytest.importorskip("torch_harmonics")

from ..spherical_swe import SphericalSWEDataset


@pytest.fixture
def dataset():
    dataset = SphericalSWEDataset.__new__(SphericalSWEDataset)
    dataset.num_examples = 3
    dataset.ictype = "random"
    dataset.normalize = False
    dataset._get_sample = lambda: (torch.ones(1), torch.zeros(1))
    return dataset


def test_getitem_accepts_valid_integer(dataset):
    sample = dataset[0]

    assert sample["x"].equal(torch.ones(1))
    assert sample["y"].equal(torch.zeros(1))


@pytest.mark.parametrize("index", ["foo", 1.5, slice(None)])
def test_getitem_rejects_non_integer_indices(dataset, index):
    with pytest.raises(TypeError, match="indices must be integers"):
        dataset[index]


@pytest.mark.parametrize("index", [-1, 3])
def test_getitem_rejects_out_of_range_indices(dataset, index):
    with pytest.raises(IndexError, match="out of range"):
        dataset[index]
