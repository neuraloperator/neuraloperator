from typing import List, Union
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dict_dataset import DictDataset
from ..transforms.normalizers import UnitGaussianNormalizer

from .ot_datamodule import OTDataModule
from .web_utils import download_from_zenodo_record
from neuralop.utils import get_project_root

class CarOTDataset(OTDataModule):
    """
    Parameters
    ----------
    root_dir : Union[str, Path]
        root directory at which data is stored.
    n_train : int, optional
        Number of training instances to load, by default 1
    n_test : int, optional
        Number of testing instances to load, by default 1
    expand_factor : float, optional
        Scale factor to map physical mesh size to latent mesh size (e.g., torus/sphere).
    reg : float, optional
        Regularization coefficient for the Sinkhorn algorithm.
    device : Union[str, torch.device], optional
        Device for OT computation.

    Attributes
    ----------
    train_loader: torch.utils.data.DataLoader
        dataloader of training examples
    test_loader: torch.utils.data.DataLoader
        dataloader of testing examples

    """

    def __init__(self,
        root_dir: Union[str, Path],
        n_train: int = 1,
        n_test: int = 1,
        expand_factor: float = 3.0, 
        reg: float = 1e-06,
        device: Union[str, torch.device] = 'cuda',
        ):

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        
        if not root_dir.exists():
            root_dir.mkdir(parents=True)

        super().__init__(
            root_dir=root_dir,
            item_dir_name='data',
            n_total=n_train+n_test,
            expand_factor=expand_factor, 
            reg=reg,
            device=device,
        )
        
        # process data list to remove specific vertices from pressure to match number of vertices
        for i, item_data in enumerate(self.data):
            press = item_data['press']
            self.data[i]['press'] = torch.cat((press[0:16], press[112:]), axis=0)
        
        # encode transport and pressure
        normalizer_keys = ['trans','press']
        self.normalizers = UnitGaussianNormalizer.from_dataset(
                self.data[0:n_train], dim=[1], keys=normalizer_keys
            )
        
        for attr in normalizer_keys:
            for j in range(len(self.data)):
                data_elem = self.data[j][attr]
                self.data[j][attr] = self.normalizers[attr].transform(data_elem)

        # prepare the complete features presenting transport map/plan on latent mesh
        for j in range(len(self.data)):
            batch_data = self.data[j]
            n_s = batch_data['trans'].shape[1]
            n_s_sqrt = int(np.sqrt(n_s))
            normal = batch_data['nor_t']
            ind_enc = batch_data['ind_enc']
            normal = normal[ind_enc]
            normal_features = torch.cross(normal , batch_data['nor_s'].reshape(-1,3), dim=1)
            trans = torch.cat((batch_data['trans'][0], batch_data['source'], normal_features), dim=1).T.reshape(9, n_s_sqrt, n_s_sqrt).unsqueeze(0)
            self.data[j]['trans']=trans

        # Datasets
        self.train_data = DictDataset(self.data[0:n_train])
        self.test_data = DictDataset(self.data[n_train:])
    
    def train_loader(self, **kwargs):
        return DataLoader(self.train_data, **kwargs)

    def test_loader(self, **kwargs):
        return DataLoader(self.test_data, **kwargs)

class load_saved_ot:
    def __init__(self,
        n_train: int = 1,
        n_test: int = 1,
        expand_factor: float = 3.0, 
        reg: float = 1e-06,
        ):
        """
        Load the saved Car-OT dataset we package along with our module.

        See `neuralop.data.datasets.ot_datamodule` for more detailed references
        """
        n_total = n_train + n_test
        data = torch.load(get_project_root() / "neuralop/data/datasets/data" / f"ot_expand{expand_factor}_reg{reg}_num{n_total}.pt")
        # process data list to remove specific vertices from pressure to match number of vertices
        for i, item_data in enumerate(data):
            press = item_data['press']
            data[i]['press'] = torch.cat((press[0:16], press[112:]), axis=0)
        
        # encode transport and pressure
        normalizer_keys = ['trans','press']
        self.normalizers = UnitGaussianNormalizer.from_dataset(
                data[0:n_train], dim=[1], keys=normalizer_keys
            )
        
        for attr in normalizer_keys:
            for j in range(len(data)):
                data_elem = data[j][attr]
                data[j][attr] = self.normalizers[attr].transform(data_elem)

        # prepare the complete features presenting transport map/plan on latent mesh
        for j in range(len(data)):
            batch_data = data[j]
            n_s = batch_data['trans'].shape[1]
            n_s_sqrt = int(np.sqrt(n_s))
            normal = batch_data['nor_t']
            ind_enc = batch_data['ind_enc']
            normal = normal[ind_enc]
            normal_features = torch.cross(normal , batch_data['nor_s'].reshape(-1,3), dim=1)
            trans = torch.cat((batch_data['trans'][0], batch_data['source'], normal_features), dim=1).T.reshape(9, n_s_sqrt, n_s_sqrt).unsqueeze(0)
            data[j]['trans']=trans

        # Datasets
        self.train_data = DictDataset(data[0:n_train])
        self.test_data = DictDataset(data[n_train:n_test+n_train])
    
    def train_loader(self, **kwargs):
        return DataLoader(self.train_data, **kwargs)

    def test_loader(self, **kwargs):
        return DataLoader(self.test_data, **kwargs)
