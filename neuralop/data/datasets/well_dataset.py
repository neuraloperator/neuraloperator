from functools import partialmethod
from pathlib import Path
from typing import List, Union, Optional

import torch

from .tensor_dataset import TensorDataset
from ..transforms.data_processors import DefaultDataProcessor
from ..transforms.normalizers import UnitGaussianNormalizer, DictUnitGaussianNormalizer

from the_well.data import WellDataset
from the_well.utils.download import well_download

class TheWellDataset:
    """__init__ _summary_
        Base Class for TheWell datasets
        
        Parameters
        ----------
        root_dir : Path
            _description_
        well_dataset_name : _type_
            _description_
        n_train : int
            _description_
        n_test : int
            _description_
        download : bool, optional
            _description_, by default True
        return_grid : bool, optional
            _description_, by default True
        """
    def __init__(self,
                 root_dir: Path, 
                 well_dataset_name,
                 download: bool=True,
                 return_grid: bool=True,

                 ):
        
        base_path = root_dir / f"datasets/{well_dataset_name}/data/"

        if download:
            for split in ['train', 'test', 'valid']:
                data_path = base_path / split
                if not data_path.exists():
                    well_download(root_dir,
                                dataset=well_dataset_name,
                                split=split,
                                )
            # Download per-variable stats.yaml directly from the_well on GitHub
            # skip for now 
        self._train_db = WellDataset(path=str(base_path / "train"),
                                        n_steps_input=1,
                                        n_steps_output=1,
                                        return_grid=return_grid,
                                        use_normalization=False)
        
        self._test_db = WellDataset(path=str(base_path / "test"),
                                        n_steps_input=1,
                                        n_steps_output=1,
                                        return_grid=return_grid,
                                        use_normalization=False)

        
        self.normalizer = UnitGaussianNormalizer()
        
        
    @property
    def train_db(self):
        return self._train_db
    
    @property
    def test_db(self):
        return self._test_db
    
    
