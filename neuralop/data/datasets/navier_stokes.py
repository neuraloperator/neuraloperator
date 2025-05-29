import logging
import os
from pathlib import Path
from typing import Union, List

from torch.utils.data import DataLoader

from .pt_dataset import PTDataset
from .web_utils import download_from_zenodo_record
from neuralop.utils import get_project_root

logger = logging.Logger(logging.root.level)

class NavierStokesDataset(PTDataset):
    """
    NavierStokesDataset stores data generated according to the 2d
    incompressible Navier-Stokes equations. Input and output are both
    2d fields with one channel of data which describes the vorticity at each point. 

    Data source: https://zenodo.org/records/12825163

    Attributes
    ----------
    train_db: torch.utils.data.Dataset of training examples
    test_db:  ""                       of test examples
    data_processor: neuralop.data.transforms.DataProcessor to process data examples
        optional, default is None
    """
    def __init__(self,
                 root_dir: Union[Path, str],
                 n_train: int,
                 n_tests: List[int],
                 batch_size: int,
                 test_batch_sizes: List[int],
                 train_resolution: int,
                 test_resolutions: int=[16,32],
                 encode_input: bool=False, 
                 encode_output: bool=True, 
                 encoding="channel-wise",
                 channel_dim=1,
                 subsampling_rate=None,
                 download: bool=True):
        
        
        """ NavierStokesDataset

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
        # convert root dir to Path
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        if not root_dir.exists():
            root_dir.mkdir(parents=True)

        # Zenodo record ID for Navier-Stokes dataset
        zenodo_record_id = "12825163"

        # List of resolutions needed for dataset object
        resolutions = set(test_resolutions + [train_resolution])

        # We store data at these resolutions on the Zenodo archive
        available_resolutions = [128, 1024]
        for res in resolutions:
            assert res in available_resolutions, f"Error: resolution {res} not available"

        # download darcy data from zenodo archive if passed
        if download:
            files_to_download = []
            already_downloaded_files = [x.name for x in root_dir.iterdir()]
            for res in resolutions:
                if f"nsforcing_train_{res}.pt" not in already_downloaded_files or \
                f"nsforcing_test_{res}.pt" not in already_downloaded_files:    
                    files_to_download.append(f"nsforcing_{res}.tgz")
            download_from_zenodo_record(record_id=zenodo_record_id,
                                        root=root_dir,
                                        files_to_download=files_to_download)
            
        # once downloaded/if files already exist, init PTDataset
        super().__init__(root_dir=root_dir,
                       n_train=n_train,
                       n_tests=n_tests,
                       dataset_name="nsforcing",
                       batch_size=batch_size,
                       test_batch_sizes=test_batch_sizes,
                       train_resolution=train_resolution,
                       test_resolutions=test_resolutions,
                       encode_input=encode_input,
                       encode_output=encode_output,
                       encoding=encoding,
                       channel_dim=channel_dim,
                       input_subsampling_rate=subsampling_rate,
                       output_subsampling_rate=subsampling_rate)

example_data_root = get_project_root() / "neuralop/datasets/data"
# load navier stokes pt for backwards compatibility
def load_navier_stokes_pt(n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    data_root = example_data_root,
    train_resolution=128,
    test_resolutions=[128],
    encode_input=False,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,
    subsampling_rate=None,
    num_workers=1):

    dataset = NavierStokesDataset(root_dir = data_root,
                           n_train=n_train,
                           n_tests=n_tests,
                           batch_size=batch_size,
                           test_batch_sizes=test_batch_sizes,
                           train_resolution=train_resolution,
                           test_resolutions=test_resolutions,
                           encode_input=encode_input,
                           encode_output=encode_output,
                           encoding=encoding,
                           channel_dim=channel_dim,
                           subsampling_rate=subsampling_rate)
    
    # return dataloaders for backwards compat
    train_loader = DataLoader(dataset.train_db,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=True,
                              persistent_workers=False,)
    
    test_loaders = {}
    for res,test_bsize in zip(test_resolutions, test_batch_sizes):
        test_loaders[res] = DataLoader(dataset.test_dbs[res],
                                       batch_size=test_bsize,
                                       shuffle=False,
                                       num_workers=num_workers,
                                       pin_memory=True,
                                       persistent_workers=False,)
    
    return train_loader, test_loaders, dataset.data_processor
