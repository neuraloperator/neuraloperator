from typing import List, Union
from pathlib import Path

import torch

from .mesh_datamodule import MeshDataModule
from .web_utils import download_from_zenodo_record

class CarCFDDataset(MeshDataModule):
    """CarCFDDataset is a processed version of the dataset introduced in 
    [1]_. We add additional manifest files to split the provided examples
    into training and testing sets, as well as remove instances that are corrupted.

    Data source: https://zenodo.org/records/13936501

    References
    ----------
    .. _[1] : 
    
    Umetani, N. and Bickel, B. (2018). "Learning three-dimensional flow for interactive 
        aerodynamic design". ACM Transactions on Graphics, 2018. 
        https://dl.acm.org/doi/10.1145/3197517.3201325.
    """

    def __init__(self,
        root_dir: Union[str, Path],
        n_train: int = 1,
        n_test: int = 1,
        query_res: List[int] = [32,32,32],
        download: bool=True
        ):
        """__init__ _summary_

        Parameters
        ----------
        root_dir : Union[str, Path]
            root directory at which data is stored.
        n_train : int, optional
            Number of training instances to load, by default 1
        n_test : int, optional
            Number of testing instances to load, by default 1
        query_res : List[int], optional
            Dimension-wise resolution of SDF query cube, by default [32,32,32]
        download : bool, optional
            Whether to download data from Zenodo, by default True
        """
        self.zenodo_record_id = "13936501"

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        
        if not root_dir.exists():
            root_dir.mkdir(parents=True)
        
        if download:
            download_from_zenodo_record(record_id=self.zenodo_record_id,
                                        root=root_dir)
        super().__init__(
            root_dir=root_dir,
            item_dir_name='',
            n_train=n_train,
            n_test=n_test,
            query_res=query_res,
            attributes='press'
        )