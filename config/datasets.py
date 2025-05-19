from typing import List, Any, Optional

from zencfg import ConfigBase

from typing import List
from zencfg import ConfigBase

class PTDatasetConfig(ConfigBase):
    folder: str
    batch_size: int
    n_train: int
    train_resolution: int
    n_tests: List[int]
    test_resolutions: List[int]
    test_batch_sizes: List[int]
    encode_input: bool = True
    encode_output: bool = True

class BurgersDatasetConfig(PTDatasetConfig):
    folder: str = 'neuralop/data/datasets/data/' 
    batch_size: int = 16
    n_train: int = 800
    test_batch_sizes: List[int] = [16]
    n_tests: List[int] = [400]
    spatial_length: int = 128
    temporal_length: int = 101
    temporal_subsample: Optional[int] = None
    encode_input: bool = False
    encode_output: bool = False
    include_endpoint: List[bool] = [True, False]

class DarcyDatasetConfig(PTDatasetConfig):
    folder: str = "data/darcy/"
    batch_size: int = 8
    n_train: int = 1000
    train_resolution: int = 16
    n_tests: List[int] = [100, 50]
    test_resolutions: List[int] = [16, 32]
    test_batch_sizes: List[int] = [16, 16]

class NavierStokesDatasetConfig(PTDatasetConfig):
    folder: str = "data/navier_stokes/"
    batch_size: int = 8
    n_train: int = 10000
    train_resolution: int = 128
    n_tests: List[int] = [2000]
    test_resolutions: List[int] = [128]
    test_batch_sizes: List[int] = [8]
