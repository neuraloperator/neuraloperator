from typing import List, Union
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from neuralop.data.transforms.data_processors import DataProcessor

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
        Number of training instances to load, by default 1.
    n_test : int, optional
        Number of testing instances to load, by default 1.
    expand_factor : float, optional
        Scale factor to map physical mesh size to latent mesh size (e.g., torus/sphere).
        Affects OT plan surjectivity: smaller values may lead to incomplete mappings, while larger values increase computational cost but improve surjectivity.
        Choose a value balancing accuracy and efficiency, by default 3.
    reg : float, optional
        Regularization coefficient for the Sinkhorn algorithm.
        Affects OT plan surjectivity: smaller values increase precision (where fewer non-surjective plans indicate higher precision) but incur higher computational cost.
        Choose a value balancing accuracy and efficiency, by default 1e-06.
    device : Union[str, torch.device], optional
        Device for OT computation.

    Attributes
    ----------
    train_loader: torch.utils.data.DataLoader
        dataloader of training examples
    test_loader: torch.utils.data.DataLoader
        dataloader of testing examples

    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        n_train: int = 1,
        n_test: int = 1,
        expand_factor: float = 3.0,
        reg: float = 1e-06,
        device: Union[str, torch.device] = "cuda",
    ):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        if not root_dir.exists():
            root_dir.mkdir(parents=True)

        super().__init__(
            root_dir=root_dir,
            item_dir_name="data/",
            n_train=n_train,
            n_test=n_test,
            attributes=["press"],
            expand_factor=expand_factor,
            reg=reg,
            device=device,
        )

        # encode transport and pressure
        normalizer_keys = ["trans", "press"]
        self.normalizers = UnitGaussianNormalizer.from_dataset(
            self.data[0:n_train], dim=[1], keys=normalizer_keys
        )

        for attr in normalizer_keys:
            for j in range(len(self.data)):
                data_elem = self.data[j][attr]
                self.data[j][attr] = self.normalizers[attr].transform(data_elem)

        for j in range(len(self.data)):
            batch_data = self.data[j]
            # prepare the complete features presenting transport map/plan on latent mesh
            n_s = batch_data["trans"].shape[1]
            n_s_sqrt = int(np.sqrt(n_s))
            normal = batch_data["nor_t"]
            ind_enc = batch_data["ind_enc"]
            normal = normal[ind_enc]
            normal_features = torch.cross(
                normal, batch_data["nor_s"].reshape(-1, 3), dim=1
            )
            trans = (
                torch.cat(
                    (batch_data["trans"][0], batch_data["source"], normal_features),
                    dim=1,
                )
                .T.reshape(9, n_s_sqrt, n_s_sqrt)
                .unsqueeze(0)
            )
            self.data[j]["trans"] = trans
            # process data list to remove specific vertices from pressure to match number of vertices
            press = batch_data["press"]
            self.data[j]["press"] = torch.cat((press[:, 0:16], press[:, 112:]), axis=1)

        # Datasets
        self.train_data = DictDataset(self.data[0:n_train])
        self.test_data = DictDataset(self.data[n_train:])

    def train_loader(self, **kwargs):
        return DataLoader(self.train_data, **kwargs)

    def test_loader(self, **kwargs):
        return DataLoader(self.test_data, **kwargs)


class load_saved_ot:
    def __init__(
        self,
        n_train: int = 2,
        n_test: int = 1,
        expand_factor: float = 3.0,
        reg: float = 1e-06,
    ):
        """
        Load the saved Car-OT dataset we package along with our module.

        See `neuralop.data.datasets.ot_datamodule` for more detailed references
        """
        data = torch.load(
            get_project_root()
            / "neuralop/data/datasets/data"
            / f"ot_expand{expand_factor}_reg{reg}_train{n_train}_test{n_test}.pt"
        )

        # encode transport and pressure
        normalizer_keys = ["trans", "press"]
        self.normalizers = UnitGaussianNormalizer.from_dataset(
            data[0:n_train], dim=[1], keys=normalizer_keys
        )

        for attr in normalizer_keys:
            for j in range(len(data)):
                data_elem = data[j][attr]
                data[j][attr] = self.normalizers[attr].transform(data_elem)

        for j in range(len(data)):
            batch_data = data[j]
            # prepare the complete features presenting transport map/plan on latent mesh
            n_s = batch_data["trans"].shape[1]
            n_s_sqrt = int(np.sqrt(n_s))
            normal = batch_data["nor_t"]
            ind_enc = batch_data["ind_enc"]
            normal = normal[ind_enc]
            normal_features = torch.cross(
                normal, batch_data["nor_s"].reshape(-1, 3), dim=1
            )
            trans = (
                torch.cat(
                    (batch_data["trans"][0], batch_data["source"], normal_features),
                    dim=1,
                )
                .T.reshape(9, n_s_sqrt, n_s_sqrt)
                .unsqueeze(0)
            )
            data[j]["trans"] = trans
            # process data list to remove specific vertices from pressure to match number of vertices
            press = batch_data["press"]
            data[j]["press"] = torch.cat((press[:, 0:16], press[:, 112:]), axis=1)

        # Datasets
        self.train_data = DictDataset(data[0:n_train])
        self.test_data = DictDataset(data[n_train : n_test + n_train])

    def train_loader(self, **kwargs):
        return DataLoader(self.train_data, **kwargs)

    def test_loader(self, **kwargs):
        return DataLoader(self.test_data, **kwargs)


# Handle data preprocessing to OTNO
class CFDDataProcessor(DataProcessor):
    """
    Implements logic to preprocess data/handle model outputs
    to train an OTNO on the CFD car-pressure dataset
    """

    def __init__(self, normalizer, device="cuda"):
        super().__init__()
        self.normalizer = normalizer
        self.device = device
        self.model = None

    def preprocess(self, sample):
        # Turn a data dictionary returned OTDataModule's DictDataset
        # into the form expected by the OTNO

        x = sample["trans"].squeeze(0).to(self.device)
        ind_dec = sample["ind_dec"].squeeze(0).to(self.device)

        # Output data
        truth = sample["press"].squeeze(0).unsqueeze(-1).to(self.device)

        batch_dict = dict(x=x, ind_dec=ind_dec, y=truth)

        sample.update(batch_dict)
        return sample

    def postprocess(self, out, sample):
        if not self.training:
            out = self.normalizer.inverse_transform(out)
            y = self.normalizer.inverse_transform(sample["y"].squeeze(0))
            sample["y"] = y
        sample = {"y": sample["y"]}
        return out, sample

    def to(self, device):
        self.device = device
        self.normalizer = self.normalizer.to(device)
        return self

    def wrap(self, model):
        self.model = model

    def forward(self, sample):
        sample = self.preprocess(sample)
        out = self.model(sample)
        out, sample = self.postprocess(out, sample)
        return out, sample
