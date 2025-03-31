from pathlib import Path
from typing import Optional, List, Union
from torch.utils.data import DataLoader

from .pt_dataset import PTDataset
from ..transforms.data_processors import DefaultDataProcessor

class Burgers1dTimeDataProcessor(DefaultDataProcessor):
    """Burgers1dTimeDataProcessor wraps the DefaultDataProcessor
    but adds one line to ``.preprocess`` to repeat the input ``x`` along
    the temporal dimension. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def preprocess(self, data_dict, batched=True):
        """preprocess does the same thing as ``DefaultDataProcessor.preprocess()``,
        with the addition of unsqueezing ``x`` along the temporal dimension and repeating
        to match ``y``'s shape. 

        Parameters
        ----------
        data_dict : dict
            one batch of input
        batched : bool, optional
            Whether inputs are batched, by default True
        """
        _, _, temporal_len, _ = data_dict["y"].shape
        # x starts as shape b, 1, spatial_len
        x = data_dict["x"]
        x = x.unsqueeze(-2).repeat([1, 1, temporal_len, 1])
        data_dict["x"] = x
        return super().preprocess(data_dict, batched)


class Burgers1dTimeDataset(PTDataset):
    """
    Burgers1dTimeDataset wraps data from the viscous 
    Burger's equation in 1 spatial dimension.
    This dataset is not available for download online, but we
    provide a low-res version on 16 spatial points

    Parameters
    ----------
    root_dir : Union[Path, str]
        root at which to download data files
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
        resolution of data for testing sets, by default [16]
    temporal_subsample : int, optional
        rate at which to subsample the temporal dimension, by default None
    spatial_subsample : int, optional
        rate at which to subsample along the spatial dimension, by default None

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
            ):
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
                         channel_dim=1,
                         dataset_name="burgers",) 
        self._data_processor = Burgers1dTimeDataProcessor(self._data_processor.in_normalizer, self.data_processor.out_normalizer)
        
def load_mini_burgers_1dtime(data_path: Union[Path, str],
                        n_train: int, 
                        n_test: int, 
                        batch_size: int, 
                        test_batch_size: int,
                        temporal_subsample: int=1,
                        spatial_subsample: int=1):
    '''
    Legacy function to load mini Burger's equation dataset
    
    Parameters
    ----------
    root_dir : Union[Path, str]
        root at which to download data files
    n_train : int
        number of train instances
    n_test : int
        number of test instances per test dataset
    batch_size : int
        batch size of training set
    test_batch_size : int
        batch size of test set
    temporal_subsample : int, optional
        rate at which to subsample the temporal dimension, by default None
    spatial_subsample : int, optional
        rate at which to subsample along the spatial dimension, by default None
    '''
    burgers_dataset = Burgers1dTimeDataset(root_dir=data_path,
                                           n_train=n_train,
                                           n_tests=[n_test],
                                           batch_size=batch_size,
                                           test_batch_sizes=[test_batch_size],
                                           train_resolution=16,
                                           test_resolutions=[16],
                                           temporal_subsample=temporal_subsample,
                                           spatial_subsample=spatial_subsample)
    train_loader = DataLoader(burgers_dataset.train_db,
                              batch_size=batch_size,
                              shuffle=True)
    
    test_loaders = {16: DataLoader(burgers_dataset.test_dbs[16],
                                   batch_size=test_batch_size,
                                   shuffle=False)}

    return train_loader, test_loaders, burgers_dataset.data_processor