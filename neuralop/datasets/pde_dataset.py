from functools import partialmethod
from pathlib import Path
from typing import List, Union

import torch

from .output_encoder import UnitGaussianNormalizer
from .tensor_dataset import TensorDataset
from .transforms import PositionalEmbedding2D
from .data_transforms import DefaultDataProcessor

class BasePDEDataset:
    """PDEDataset is the base Dataset class for our library.
            Datasets contain input-output pairs a(x), u(x) and may also
            contain additional information, e.g. function parameters,
            input geometry or output query points.

            datasets may implement a download flag at init, which provides
            access to a number of premade datasets for sample problems provided
            in our Zenodo archive. 

        All datasets are required to expose the following attributes after init:

        train_db: torch.utils.data.Dataset of training examples
        test_db:  ""                       of test examples
        train_loader: torch.utils.data.DataLoader for single-node, single-GPU training
        test_loaders: torch.utils.data.DataLoader (s) for single-node, single-GPU training
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
            download_zenodo_dataset(root_dir=root_dir,
                                    dataset_name=dataset_name,
                                    train_resolutions=[train_resolution],
                                    test_resolutions=test_resolutions)
            
        
        # Load train data
        data = torch.load(
        Path(root_dir).joinpath(f"{dataset_name}_train_{train_resolution}.pt").as_posix()
        )
        x_train = (
        data["x"][0:n_train, ...].unsqueeze(channel_dim).type(torch.float32).clone()
        )
        y_train = data["y"][0:n_train, ...].unsqueeze(channel_dim).clone()
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
            pos_encoding = PositionalEmbedding2D(grid_boundaries=grid_boundaries)
        else:
            pos_encoding = None

        self._data_processor = DefaultDataProcessor(in_normalizer=input_encoder,
                                                   out_normalizer=output_encoder,
                                                   positional_encoding=pos_encoding)

        # load test data
        self._test_dbs = {}
        for (res, n_test) in zip(
        test_resolutions, n_tests
        ):
            print(
                f"Loading test db at resolution {res} with {n_test} samples "
            )
            data = torch.load(Path(root_dir).joinpath(f"{dataset_name}_test_{res}.pt").as_posix())
            x_test = (
                data["x"][:n_test, ...].unsqueeze(channel_dim).type(torch.float32).clone()
            )
            y_test = data["y"][:n_test, ...].unsqueeze(channel_dim).clone()
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
    
def download_zenodo_dataset(dataset_name: str,
                train_resolutions: List[int],
                test_resolutions: List[int],
                ):
    data_files = [f"{dataset_name}_train_{res}.pt" for res in train_resolutions]
    data_files += [f"{dataset_name}_test_{res}.pt" for res in test_resolutions]
    
    # For now this is a placeholder
    print(f"Downloading {data_files}")

def partialclass(new_name, cls, *args, **kwargs):
    """Create a new class with different default values

    Along the lines of neuralop.models.base_model
    """
    __init__ = partialmethod(cls.__init__, *args, **kwargs)
    new_class = type(
        new_name,
        (cls,),
        {
            "__init__": __init__,
            "__doc__": cls.__doc__,
        },
    )
    return new_class

DarcyFlowDataset = partialclass("DarcyFlowDataset", BasePDEDataset, dataset_name='darcy')
NavierStokesDataset = partialclass("NavierStokesDataset", BasePDEDataset, dataset_name='nsforcing')
