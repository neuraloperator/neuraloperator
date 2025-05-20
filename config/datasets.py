from typing import List, Any, Optional

from zencfg import ConfigBase

from typing import List
from zencfg import ConfigBase

class PTDatasetConfig(ConfigBase):
    """PTDatasetConfig provides configuration options
    for the PTDataset base datasets we package with the library.

    Parameters
    ----------
    folder: str
        root path of dataset
    batch_size: int
        training batch size
    n_train: int
        number of training examples
    train_resolution: int
        resolution of training data
    n_tests: List[int]
        List of numbers of test examples. These should
        correspond 1-to-1 with ``test_resolutions``
        and ``test_batch_sizes``.
    test_resolutions: List[int]
        List of numbers of test resolutions. These should
        correspond 1-to-1 with ``n_tests`` and ``test_batch_sizes``.
    test_batch_sizes: List[int]
        List of batch sizes for each test set. These should
        correspond 1-to-1 with ``n_tests`` and ``test_resolutions``.
    encode_input: bool = True
        whether to apply a standardization (e.g. via 
        Gaussian normalization) to input. 
    encode_output: bool = True
        whether to inverse above standardization (e.g. via 
        Gaussian normalization) to outputs **during training only**. 
    """
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
    """BurgersDatasetConfig provides configuration options
    for the Burgers1dTimeDataset module we package with the library.

    Parameters
    ----------
    folder: str
        root path of dataset
    batch_size: int
        training batch size
    n_train: int
        number of training examples
    train_resolution: int
        resolution of training data
    n_tests: List[int]
        List of numbers of test examples. These should
        correspond 1-to-1 with ``test_resolutions``
        and ``test_batch_sizes``.
    test_resolutions: List[int]
        List of numbers of test resolutions. These should
        correspond 1-to-1 with ``n_tests`` and ``test_batch_sizes``.
    test_batch_sizes: List[int]
        List of batch sizes for each test set. These should
        correspond 1-to-1 with ``n_tests`` and ``test_resolutions``.
    encode_input: bool = True
        whether to apply a standardization (e.g. via 
        Gaussian normalization) to input. 
    encode_output: bool = True
        whether to inverse above standardization (e.g. via 
        Gaussian normalization) to outputs **during training only**. 
    """
    folder: str = 'neuralop/data/datasets/data/' 
    batch_size: int = 16
    n_train: int = 800
    test_batch_sizes: List[int] = [16]
    n_tests: List[int] = [400]
    # full res is 128x101. We redistribute a mini version at 16x17
    spatial_length: int = 16 
    temporal_length: int = 17
    temporal_subsample: Optional[int] = None
    encode_input: bool = False
    encode_output: bool = False
    include_endpoint: List[bool] = [True, False]

class DarcyDatasetConfig(PTDatasetConfig):
    """Provides configuration options
    for the DarcyDataset module we package with the library.

    Parameters
    ----------
    folder: str
        root path of dataset
    batch_size: int
        training batch size
    n_train: int
        number of training examples
    train_resolution: int
        resolution of training data
    n_tests: List[int]
        List of numbers of test examples. These should
        correspond 1-to-1 with ``test_resolutions``
        and ``test_batch_sizes``.
    test_resolutions: List[int]
        List of numbers of test resolutions. These should
        correspond 1-to-1 with ``n_tests`` and ``test_batch_sizes``.
    test_batch_sizes: List[int]
        List of batch sizes for each test set. These should
        correspond 1-to-1 with ``n_tests`` and ``test_resolutions``.
    encode_input: bool = True
        whether to apply a standardization (e.g. via 
        Gaussian normalization) to input. 
    encode_output: bool = True
        whether to inverse above standardization (e.g. via 
        Gaussian normalization) to outputs **during training only**. 
    """
    folder: str = "data/darcy/"
    batch_size: int = 8
    n_train: int = 1000
    train_resolution: int = 16
    n_tests: List[int] = [100, 50]
    test_resolutions: List[int] = [16, 32]
    test_batch_sizes: List[int] = [16, 16]

class NavierStokesDatasetConfig(PTDatasetConfig):
    """NavierStokesDatasetConfig provides configuration options
    for the NavierStokesDataset module we package with the library.

    Parameters
    ----------
    folder: str
        root path of dataset
    batch_size: int
        training batch size
    n_train: int
        number of training examples
    train_resolution: int
        resolution of training data
    n_tests: List[int]
        List of numbers of test examples. These should
        correspond 1-to-1 with ``test_resolutions``
        and ``test_batch_sizes``.
    test_resolutions: List[int]
        List of numbers of test resolutions. These should
        correspond 1-to-1 with ``n_tests`` and ``test_batch_sizes``.
    test_batch_sizes: List[int]
        List of batch sizes for each test set. These should
        correspond 1-to-1 with ``n_tests`` and ``test_resolutions``.
    encode_input: bool = True
        whether to apply a standardization (e.g. via 
        Gaussian normalization) to input. 
    encode_output: bool = True
        whether to inverse above standardization (e.g. via 
        Gaussian normalization) to outputs **during training only**. 
    """
    folder: str = "data/navier_stokes/"
    batch_size: int = 8
    n_train: int = 10000
    train_resolution: int = 128
    n_tests: List[int] = [2000]
    test_resolutions: List[int] = [128]
    test_batch_sizes: List[int] = [8]

class CarCFDDatasetConfig(ConfigBase):
    root: str
    sdf_query_resolution: int
    n_train: int = 500
    n_test: int = 111
    download: bool = True
