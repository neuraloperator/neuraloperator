from ..burgers import Burgers1dTimeDataset
from ..darcy import DarcyDataset
from ..navier_stokes import NavierStokesDataset
from pathlib import Path

import os
import shutil

import pytest

test_data_dir = Path("./dataset_test")

@pytest.mark.parametrize('resolution', [16])
def test_DarcyDatasetDownload(resolution):
    dataset = DarcyDataset(root_dir=test_data_dir,
                           n_train=5,
                           n_tests=[5],
                           batch_size=1,
                           test_batch_sizes=[1],
                           train_resolution=resolution,
                           test_resolutions=[resolution, resolution*2],
                           download=True)
    
    downloaded_files = os.listdir(test_data_dir)
    for split in ['train','test']:
        for res in [resolution, resolution*2]:
            assert f"darcy_{split}_{res}.pt" in downloaded_files
    assert dataset.train_db
    assert dataset.test_dbs
    assert dataset.data_processor
    shutil.rmtree(test_data_dir)

burgers_root = Path("./neuralop/data/datasets/data/").resolve()

@pytest.mark.parametrize('resolution', [16])
def test_BurgersDataset(resolution):
    dataset = Burgers1dTimeDataset(
                        root_dir=burgers_root,
                        n_train=5,
                        n_tests=[5],
                        batch_size=1,
                        test_batch_sizes=[1],
                        train_resolution=resolution,
                        test_resolutions=[resolution])
    
    assert dataset.train_db
    assert dataset.test_dbs
    assert dataset.data_processor
    
def test_NSDownload():
    # monkeypatch bypasses confirmation input
    dataset = NavierStokesDataset(root_dir=test_data_dir,
                           n_train=5,
                           n_tests=[5],
                           batch_size=1,
                           test_batch_sizes=[1],
                           train_resolution=128,
                           test_resolutions=[128],
                           download=True)
    
    downloaded_files = os.listdir(test_data_dir)
    for split in ['train','test']:
        for res in [128]:
            assert f"nsforcing_{split}_{res}.pt" in downloaded_files
    assert dataset.train_db
    assert dataset.test_dbs
    assert dataset.data_processor
    shutil.rmtree(test_data_dir)