from functools import partial
from pathlib import Path
from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader

from .output_encoder import UnitGaussianNormalizer
from .tensor_dataset import TensorDataset
from .transforms import PositionalEmbedding2D
from .data_transforms import DataProcessor, DefaultDataProcessor
from .pde_dataset import PDEDataset, download_fn

class DarcyFlowDataset(PDEDataset):
    def __init__(self,
                 root_dir: Union[Path, str],
                 n_train: int,
                 n_tests: List[int],
                 batch_size: int,
                 test_batch_sizes: List[int],
                 train_resolution: int,
                 test_resolutions: int=[16,32],
                 grid_boundaries: List[int]=[[0,1],[0,1]],
                 positional_encoding: bool=True,
                 encode_input: bool=False, 
                 encode_output: bool=True, 
                 encoding="channel-wise",
                 channel_dim=1,
                 download: bool=True):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        
        self.root_dir = root_dir

        # save dataloader properties for later
        self.batch_size = batch_size
        self.test_resolutions = test_resolutions
        self.test_batch_sizes = test_batch_sizes

        # Download data if it isn't already downloaded
        if download:
            if not root_dir.exists():
                root_dir.mkdir(parents=True)

            # iterate through all required filenames and download
            # if the files need to be downloaded
            required_fnames = [f"darcy_train_{train_resolution}"] + \
                    [f"darcy_test_{res}.pt" for res in test_resolutions]
            for fname in required_fnames:
                fpath = root_dir / fname
                if not fpath.exists():
                    download_fn(fname)
        
        # Load train data
        data = torch.load(
        Path(root_dir).joinpath(f"darcy_train_{train_resolution}.pt").as_posix()
        )
        x_train = (
        data["x"][0:n_train, :, :].unsqueeze(channel_dim).type(torch.float32).clone()
        )
        y_train = data["y"][0:n_train, :, :].unsqueeze(channel_dim).clone()
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
        self.train_db = TensorDataset( 
            x_train,
            y_train,
        )

        # create pos encoder and DataProcessor
        if positional_encoding:
            pos_encoding = PositionalEmbedding2D(grid_boundaries=grid_boundaries)
        else:
            pos_encoding = None

        self._data_processor = DefaultDataProcessor(in_normalizer=input_encoder,
                                                   out_normalizer=output_encoder,
                                                   positional_encoding=pos_encoding)

        # load test data
        self.test_dbs = {}
        for (res, n_test, test_batch_size) in zip(
        test_resolutions, n_tests, test_batch_sizes
        ):
            print(
                f"Loading test db at resolution {res} with {n_test} samples "
            )
            data = torch.load(Path(root_dir).joinpath(f"darcy_test_{res}.pt").as_posix())
            x_test = (
                data["x"][:n_test, :, :].unsqueeze(channel_dim).type(torch.float32).clone()
            )
            y_test = data["y"][:n_test, :, :].unsqueeze(channel_dim).clone()
            del data

            test_db = TensorDataset(
                x_test,
                y_test,
            )
            self.test_dbs[res] = test_db

    def data_processor(self) -> DataProcessor:
        return self._data_processor
    
    def train_loader(self, 
                     num_workers: int=None, 
                     pin_memory: bool=True,
                     persistent_workers: bool=False) -> DataLoader:
        
        return DataLoader(dataset=self.train_db,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            persistent_workers=persistent_workers,
                            )
    
    def test_loaders(self, 
                     num_workers: int=None, 
                     pin_memory: bool=True,
                     persistent_workers: bool=False) -> Dict[DataLoader]:
        test_loaders = {}
        for (res, batch_size) in zip(self.test_resolutions, self.test_batch_sizes):
            loader = DataLoader(dataset=self.test_dbs[res],
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                persistent_workers=persistent_workers,
                                )
            test_loaders[res] = loader
        return test_loaders        

# Load small darcy flow as a partial class of DarcyFlowDataset
SmallDarcyFlowDataset = partial(DarcyFlowDataset, train_resolution=16)