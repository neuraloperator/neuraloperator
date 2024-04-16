
import torch
from pathlib import Path
from typing import Union, List

from .pt_dataset import PTDataset
from .web_utils import download_from_url

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

        # url/md5 info for data hosted on Zenodo archive
        dataset_info = {
            16: {
                'train': {
                    "url": "https://zenodo.org/records/10982484/files/darcy_train_16.pt?download=1",
                    "md5": "248e3c55c8c4b5a41ff2b972bf56c7d9"
                },
                'test': {
                    "url": "https://zenodo.org/records/10982484/files/darcy_test_16.pt?download=1",
                    "md5": "9f747d431dc5fd91b5bff2dd580ae452"
                },
            }
        }

        # download darcy data from zenodo archive if passed
        if download:
            train_info = dataset_info[train_resolution]["train"]
            download_from_url(url=train_info["url"], 
                              md5=train_info["md5"],
                              root=root_dir,
                              filename=f"darcy_train_{train_resolution}.pt")
            for test_res in test_resolutions:
                test_info = dataset_info[test_res]["test"]
                download_from_url(url=test_info["url"], 
                                md5=test_info["md5"],
                                root=root_dir,
                                filename=f"darcy_test_{test_res}.pt")
        
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