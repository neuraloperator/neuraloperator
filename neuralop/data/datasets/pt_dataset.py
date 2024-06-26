from functools import partialmethod
from pathlib import Path
from typing import List, Union, Optional

import torch

from .tensor_dataset import TensorDataset
from ..transforms.data_processors import DefaultDataProcessor
from ..transforms.normalizers import UnitGaussianNormalizer
from neuralop.layers.embeddings import GridEmbedding2D

class PTDataset:
    """PTDataset is a base Dataset class for our library.
            PTDatasets contain input-output pairs a(x), u(x) and may also
            contain additional information, e.g. function parameters,
            input geometry or output query points.

            datasets may implement a download flag at init, which provides
            access to a number of premade datasets for sample problems provided
            in our Zenodo archive. 

        All datasets are required to expose the following attributes after init:

        train_db: torch.utils.data.Dataset of training examples
        test_db:  ""                       of test examples
        data_processor: neuralop.datasets.DataProcessor to process data examples
            optional, default is None
        """
    def __init__(self,
                 root_dir: Union[Path, str],
                 dataset_name: str,
                 n_train: int,
                 n_tests: List[int],
                 batch_size: int,
                 test_batch_sizes: List[int],
                 train_resolution: int,
                 test_resolutions: List[int]=[16,32],
                 grid_boundaries: List[List[int]]=[[0,1],[0,1]],
                 positional_encoding: bool=True,
                 encode_input: bool=False, 
                 encode_output: bool=True, 
                 encoding="channel-wise",
                 subsampling_rate: Optional[Union[List[int],int]]=None,
                 channel_dim=1,):
        """PTDataset _summary_

        Parameters
        ----------
        root_dir : Union[Path, str]
            _description_
        dataset_name : str
            _description_
        n_train : int
            _description_
        n_tests : List[int]
            _description_
        batch_size : int
            _description_
        test_batch_sizes : List[int]
            _description_
        train_resolution : int
            _description_
        test_resolutions : List[int], optional
            _description_, by default [16,32]
        grid_boundaries : List[List[int]], optional
            _description_, by default [[0,1],[0,1]]
        positional_encoding : bool, optional
            _description_, by default True
        encode_input : bool, optional
            _description_, by default False
        encode_output : bool, optional
            _description_, by default True
        encoding : str, optional
            _description_, by default "channel-wise"
        subsampling_rate : Optional[Union[List[int],int]], optional
            _description_, by default None
        channel_dim : int, optional
            _description_, by default 1
        """
        
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        
        self.root_dir = root_dir

        # save dataloader properties for later
        self.batch_size = batch_size
        self.test_resolutions = test_resolutions
        self.test_batch_sizes = test_batch_sizes
            
        # Load train data
        
        data = torch.load(
        Path(root_dir).joinpath(f"{dataset_name}_train_{train_resolution}.pt").as_posix()
        )
        # optionally subsample along data indices
        data_dims = data["x"].ndim - 1
        # convert None and 0 to 1
        if not subsampling_rate:
            subsampling_rate = 1
        if not isinstance(subsampling_rate, list):
            # expand subsampling rate along dims if one per dim is not provided
            subsampling_rate = [subsampling_rate] * data_dims
        # make sure there is one subsampling rate per data dim
        assert len(subsampling_rate) == data_dims
        
        train_indices = [slice(0, n_train, None)] + [slice(None, None, rate) for rate in subsampling_rate]
        x_train = (
        data["x"][train_indices].unsqueeze(channel_dim).type(torch.float32).clone()
        )
        y_train = data["y"][train_indices].unsqueeze(channel_dim).clone()
        del data

        # Fit optional encoders to train data
        # Actual encoding happens within DataProcessor
        if encode_input:
            if encoding == "channel-wise":
                reduce_dims = list(range(x_train.ndim))
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            input_encoder.fit(x_train)
        else:
            input_encoder = None

        if encode_output:
            if encoding == "channel-wise":
                reduce_dims = list(range(y_train.ndim))
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            output_encoder.fit(y_train)
        else:
            output_encoder = None

        # Save train dataset
        self._train_db = TensorDataset( 
            x_train,
            y_train,
        )

        # create pos encoder and DataProcessor
        if positional_encoding:
            pos_encoding = GridEmbedding2D(grid_boundaries=grid_boundaries)
        else:
            pos_encoding = None

        self._data_processor = DefaultDataProcessor(in_normalizer=input_encoder,
                                                   out_normalizer=output_encoder,
                                                   positional_encoding=pos_encoding)

        # load test data
        self._test_dbs = {}
        for (res, n_test) in zip(test_resolutions, n_tests):
            print(
                f"Loading test db for resolution {res} with {n_test} samples "
            )
            data = torch.load(Path(root_dir).joinpath(f"{dataset_name}_test_{res}.pt").as_posix())

            # optionally subsample along data indices
            test_indices = [slice(0, n_test, None)] + [slice(None, None, rate) for rate in subsampling_rate] 
            x_test = (
                data["x"][test_indices].unsqueeze(channel_dim).type(torch.float32).clone()
            )
            y_test = data["y"][test_indices].unsqueeze(channel_dim).clone()
            del data

            test_db = TensorDataset(
                x_test,
                y_test,
            )
            self._test_dbs[res] = test_db
    
    @property
    def data_processor(self):
        return self._data_processor
    
    @property
    def train_db(self):
        return self._train_db
    
    @property
    def test_dbs(self):
        return self._test_dbs