import sys

from neuralop.datasets.cfd_dataset import CFDDataset, load_cfd


def test_cfd():
    # python -m pytest -s neuralop/datasets/tests/test_cfd.py PATH_TO_CFD_DATA
    # Test the dataset
    path = sys.argv[-1]
    dataset = CFDDataset(path)
    print(len(dataset))
    train_loader, test_loader, pressure_normalization = load_cfd(
        data_path=path, n_train=500, n_test=111, batch_size=32, test_batch_size=32
    )
    train_iter = iter(train_loader)
    for i in range(10):
        data = next(train_iter)
        assert len(data) == 2
        print(data[0].shape, data[1].shape)
        assert data[0].shape[0] == 32
        assert data[1].shape[0] == 32
        assert data[0].shape[1] == data[1].shape[1]
    test_iter = iter(test_loader)
    for i in range(2):
        data = next(test_iter)
        assert len(data) == 2
        print(data[0].shape, data[1].shape)
        assert data[0].shape[0] == 32
        assert data[1].shape[0] == 32
        assert data[0].shape[1] == data[1].shape[1]
