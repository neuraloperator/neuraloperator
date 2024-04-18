
import logging
from pathlib import Path
from typing import Union, List

from .pt_dataset import PTDataset
from .web_utils import download_from_zenodo_record

logger = logging.Logger(logging.root.level)

class DarcyDataset(PTDataset):
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
        
        zenodo_record_id = "10982484"
        resolutions = set(test_resolutions + [train_resolution])
        available_resolutions = [16, 32, 64, 128, 421]
        for res in resolutions:
            assert res in available_resolutions, f"Error: resolution {res} not available"

        # download darcy data from zenodo archive if passed
        if download:
            files_to_download = [f"darcy_{res}.tgz" for res in resolutions]
            download_from_zenodo_record(record_id=zenodo_record_id,
                                        root=root_dir,
                                        files_to_download=files_to_download)
            
        # once downloaded/if files already exist, init PTDataset
        super.__init__(root_dir=root_dir,
                       dataset_name="darcy",
                       n_train=n_train,
                       n_tests=n_tests,
                       batch_size=batch_size,
                       test_batch_sizes=test_batch_sizes,
                       train_resolution=train_resolution,
                       test_resolutions=test_resolutions,
                       grid_boundaries=grid_boundaries,
                       positional_encoding=positional_encoding,
                       encode_input=encode_input,
                       encode_output=encode_output,
                       encoding=encoding,
                       channel_dim=channel_dim,)