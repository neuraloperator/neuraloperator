from functools import partialmethod
from pathlib import Path
from typing import List, Union, Optional

import torch

from .tensor_dataset import TensorDataset
from ..transforms.data_processors import DefaultDataProcessor
from ..transforms.normalizers import UnitGaussianNormalizer

from the_well.data import WellDataset
from the_well.utils.download import well_download

class TheWellDataset:
    def __init__(root_dir: Path, 
                 well_dataset_name,
                 n_train: int,
                 n_test: int, 
                 download: bool=True,
                 return_grid: bool=True,

                 ):
        
        """ Base Class for TheWell datasets
        """
        base_path = root_dir / f"datasets/{well_dataset_name}"

        if download:
            for split in ['train', 'test']:
                data_path = base_path / f"data/{split}"
                if not data_path.exists():
                    well_download(base_path=root_dir,
                                dataset=well_dataset_name,
                                split=split,
                                )
        
        dataset = WellDataset(path=base_path,
                              n_steps_input=1,
                              n_steps_output=1,
                              return_grid=return_grid,
                              )
