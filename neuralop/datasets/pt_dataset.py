import torch

from ..utils import UnitGaussianNormalizer
from .tensor_dataset import GeneralTensorDataset
from .transforms import PositionalEmbedding


def load_pt_traintestsplit(data_path, 
                        n_train, n_test,
                        batch_size, test_batch_size,
                        labels='x',
                        grid_boundaries=[[0,1],[0,1]],
                        positional_encoding=True,
                        gaussian_norm=False,
                        norm_type='channel-wise', 
                        channel_dim=1,
                        subsample_fact=None,
                        interp_res=None
                        ):
    """Create train-test split from a single file
    containing any number of tensors. n_train or
    n_test can be zero. First n_train
    points are used for the training set and n_test of
    the remaining points are used for the test set.
    If subsampling or interpolation is used, all tensors 
    are assumed to be of the same dimension and the 
    operation will be applied to all.

    Parameters
    ----------
    n_train : int
    n_test : int
    batch_size : int
    test_batch_size : int
    labels: str list, default is 'x'
        tensor labels in the data file
    grid_boundaries : int list, default is [[0,1],[0,1]],
    positional_encoding : bool list, default is True
    gaussian_norm : bool list, default is False
    norm_type : str, default is 'channel-wise'
    channel_dim : int list, default is 1
        where to put the channel dimension, defaults size is batch, channel, height, width
    subsample_fact : list, default is None
    interp_res : list, default is None

    Returns
    -------
    train_loader, test_loader, encoders

    train_loader : torch DataLoader None
    test_loader : torch DataLoader None
    encoders : UnitGaussianNormalizer List[UnitGaussianNormalizer] None
    """
    data = torch.load(data_path)

    if type(labels) is not list and type(labels) is not tuple:
        labels = [labels]
        n_tensors = 1
    else:
        n_tensors = len(labels)
    
    if type(positional_encoding) is not list and type(positional_encoding) is not tuple:
        positional_encoding = [positional_encoding]*n_tensors
    
    if type(channel_dim) is not list and type(channel_dim) is not tuple:
        channel_dim = [channel_dim]*n_tensors
    
    if type(gaussian_norm) is not list and type(gaussian_norm) is not tuple:
        gaussian_norm = [gaussian_norm]*n_tensors
    
    if type(norm_type) is not list and type(norm_type) is not tuple:
        norm_type = [norm_type]*n_tensors
    
    if subsample_fact is not None:
        assert len(subsample_fact) == 2 or len(subsample_fact) == 3, "Only 2D and 3D data supported for subsampling"
    
    if interp_res is not None:
        assert len(interp_res) == 2 or len(interp_res) == 3, "Only 2D and 3D data supported for interpolation"
        if len(interp_res) == 2:
            interp_mode = 'bilinear'
            antialias = True
        else:
            interp_mode = 'trilinear'
            antialias = False
     
    if gaussian_norm[0]:
        assert n_train > 0, "Cannot normalize test data without train data"

    train_data = None
    if n_train > 0:
        train_data = []
        train_transforms = []
        for j in range(n_tensors):
            current_data = data[labels[j]][0:n_train, ...].type(torch.float32).clone()

            if channel_dim[j] is not None:
                current_data = current_data.unsqueeze(channel_dim[j])

            if subsample_fact is not None:
                if len(subsample_fact) == 2:
                    current_data = current_data[..., ::subsample_fact[0], ::subsample_fact[1]]
                else:
                    current_data = current_data[..., ::subsample_fact[0], ::subsample_fact[1], ::subsample_fact[2]]
            
            if interp_res is not None:
                current_data = torch.nn.functional.interpolate(current_data, size=interp_res, mode=interp_mode, align_corners=False, antialias=antialias)
            
            train_data.append(current_data.contiguous())

            transform = PositionalEmbedding(grid_boundaries, 0) if positional_encoding[j] else None
            train_transforms.append(transform)

    test_data = None
    if n_test > 0:
        test_data = []
        test_transforms = []
        for j in range(n_tensors):
            current_data = data[labels[j]][n_train:(n_train + n_test), ...].type(torch.float32).clone()

            if channel_dim[j] is not None:
                current_data = current_data.unsqueeze(channel_dim)

            if subsample_fact is not None:
                if len(subsample_fact) == 2:
                    current_data = current_data[..., ::subsample_fact[0], ::subsample_fact[1]]
                else:
                    current_data = current_data[..., ::subsample_fact[0], ::subsample_fact[1], ::subsample_fact[2]]
            
            if interp_res is not None:
                current_data = torch.nn.functional.interpolate(current_data, size=interp_res, mode=interp_mode, align_corners=False, antialias=antialias)
            
            test_data.append(current_data.contiguous())

            transform = PositionalEmbedding(grid_boundaries, 0) if positional_encoding[j] else None
            test_transforms.append(transform)

    del data

    encoders = []
    for j in range(n_tensors):
        if gaussian_norm[j]:
            if norm_type[j] == 'channel-wise':
                reduce_dims = list(range(train_data[j].ndim))
            else:
                reduce_dims = [0]
            
            encoder = UnitGaussianNormalizer(train_data[j], reduce_dim=reduce_dims)
            train_data[j] = encoder.encode(train_data[j].contiguous())
            if test_data is not None:
                test_data[j] = encoder.encode(test_data[j].contiguous())
            
            encoders.append(encoder)
    
    if len(encoders) == 0:
        encoders = None
    elif len(encoder) == 1:
        encoders = encoders[0]


    if train_data is not None:
        train_db = GeneralTensorDataset(train_data, train_transforms)
        train_loader = torch.utils.data.DataLoader(train_db,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=0, pin_memory=True, persistent_workers=False)
    else:
        train_loader = None

    if test_data is not None:
        test_db = GeneralTensorDataset(test_data, test_transforms)
        test_loader = torch.utils.data.DataLoader(test_db,
                                                batch_size=test_batch_size, shuffle=False,
                                                num_workers=0, pin_memory=True, persistent_workers=False)
    else:
        test_loader = None

        
    return train_loader, test_loader, encoders