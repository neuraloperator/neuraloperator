import torch
from pathlib import Path

from ..utils import UnitGaussianNormalizer
from .tensor_dataset import TensorDataset
from .transforms import PositionalEmbedding


def load_darcy_pt(data_path, 
                n_train, n_tests,
                batch_size, test_batch_sizes,
                test_resolutions=[32],
                train_resolution=32,
                grid_boundaries=[[0,1],[0,1]],
                positional_encoding=True,
                encode_input=False,
                encode_output=True,
                encoding='channel-wise', 
                channel_dim=1):
    """Load the Navier-Stokes dataset
    """
    data = torch.load(Path(data_path).joinpath(f'darcy_train_{train_resolution}.pt').as_posix())
    x_train = data['x'][0:n_train, :, :].unsqueeze(channel_dim)
    y_train = data['y'][0:n_train, :, :].unsqueeze(channel_dim)
    del data

    idx = test_resolutions.index(train_resolution)
    test_resolutions.pop(idx)
    n_test = n_tests.pop(idx)
    test_batch_size = test_batch_sizes.pop(idx)

    data = torch.load(Path(data_path).joinpath(f'darcy_test_{train_resolution}.pt').as_posix())
    x_test = data['x'][:n_test, :, :].unsqueeze(channel_dim)
    y_test = data['y'][:n_test, :, :].unsqueeze(channel_dim)
    del data
    
    if encode_input:
        if encoding == 'channel-wise':
            reduce_dims = list(range(x_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        input_encoder = UnitGaussianNormalizer(x_train, reduce_dim=reduce_dims)
        x_train = input_encoder.encode(x_train)
        x_test = input_encoder.encode(x_test.contiguous())
    else:
        input_encoder = None

    if encode_output:
        if encoding == 'channel-wise':
            reduce_dims = list(range(y_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(y_train, reduce_dim=reduce_dims)
        y_train = output_encoder.encode(y_train)
    else:
        output_encoder = None

    train_db = TensorDataset(x_train, y_train, transform_x=PositionalEmbedding(grid_boundaries, 0) if positional_encoding else None)
    train_loader = torch.utils.data.DataLoader(train_db,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=0, pin_memory=True, persistent_workers=False)

    test_db = TensorDataset(x_test, y_test,transform_x=PositionalEmbedding(grid_boundaries, 0) if positional_encoding else None)
    test_loader = torch.utils.data.DataLoader(test_db,
                                              batch_size=test_batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True, persistent_workers=False)
    test_loaders =  {train_resolution: test_loader}
    for (res, n_test, test_batch_size) in zip(test_resolutions, n_tests, test_batch_sizes):
        print(f'Loading test db at resolution {res} with {n_test} samples and batch-size={test_batch_size}')
        data = torch.load(Path(data_path).joinpath(f'darcy_test_{res}.pt').as_posix())
        x_test = data['x'][:n_test, :, :].unsqueeze(channel_dim)
        y_test = data['y'][:n_test, :, :].unsqueeze(channel_dim)
        if input_encoder is not None:
            x_test = input_encoder.encode(x_test)

        test_db = TensorDataset(x_test, y_test, transform_x=PositionalEmbedding(grid_boundaries, 0) if positional_encoding else None)
        test_loader = torch.utils.data.DataLoader(test_db,
                                                  batch_size=test_batch_size, shuffle=False,
                                                  num_workers=0, pin_memory=True, persistent_workers=False)
        test_loaders[res] = test_loader

    return train_loader, test_loaders, output_encoder
