from pathlib import Path
from typing import Optional, List, Union

import numpy as np
import torch
from .tensor_dataset import TensorDataset
from .pt_dataset import PTDataset

class Burgers1dTimeDataset(PTDataset):
    """
    Burgers1dTimeDataset wraps data from the viscous 
    Burger's equation in 1 spatial dimension.
    This dataset is not available for download online, but we
    provide a low-res version on 16 spatial points

    Attributes
    ----------
    train_db: torch.utils.data.Dataset of training examples
    test_db:  ""                       of test examples
    data_processor: neuralop.data.transforms.DataProcessor to process data examples
        optional, default is None
    """
    def __init__(
            self,
            root_dir: Union[Path, str], 
            n_train: int, 
            n_tests: list[int], 
            train_resolution: int=16,
            test_resolutions: List[int]=[16],
            batch_size: int=32, 
            test_batch_sizes: List[int]=32,
            temporal_subsample: Optional[int]=None, 
            spatial_subsample: Optional[int]=None, 
            pad: int=0,
            channel_dim: int=1,
            download: bool=True,
            ):
        """
        Parameters
        ----------
        root_dir : Union[Path, str]
            root at which to download data files
        dataset_name : str
            prefix of pt data files to store/access
        n_train : int
            number of train instances
        n_tests : List[int]
            number of test instances per test dataset
        batch_size : int
            batch size of training set
        test_batch_sizes : List[int]
            batch size of test sets
        train_resolution : int
            resolution of data for training set
        test_resolutions : List[int], optional
            resolution of data for testing sets, by default [16,32]
        encode_input : bool, optional
            whether to normalize inputs in provided DataProcessor,
            by default False
        encode_output : bool, optional
            whether to normalize outputs in provided DataProcessor,
            by default True
        encoding : str, optional
            parameter for input/output normalization. Whether
            to normalize by channel ("channel-wise") or 
            by pixel ("pixel-wise"), default "channel-wise"
        input_subsampling_rate : int or List[int], optional
            rate at which to subsample each input dimension, by default None
        output_subsampling_rate : int or List[int], optional
            rate at which to subsample each output dimension, by default None
        channel_dim : int, optional
            dimension of saved tensors to index data channels, by default 1
        """
        # convert root dir to path
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        if not root_dir.exists():
            root_dir.mkdir(parents=True)
        
        available_resolutions = [16, 128]
        assert train_resolution in available_resolutions, f"Resolutions available: {available_resolutions}, got {train_resolution}"
        for res in test_resolutions:
            assert res in available_resolutions, f"Resolutions available: {available_resolutions}, got {res}"

        super().__init__(root_dir=root_dir,
                         n_train=n_train,
                         n_tests=n_tests,
                         batch_size=batch_size,
                         test_batch_sizes=test_batch_sizes,
                         train_resolution=train_resolution,
                         test_resolutions=test_resolutions,
                         input_subsampling_rate=spatial_subsample,
                         output_subsampling_rate=[temporal_subsample, spatial_subsample],
                         encode_input=True,
                         encode_output=True,
                         encoding="channel-wise",
                         channel_dim=channel_dim,
                         dataset_name="burgers") 
