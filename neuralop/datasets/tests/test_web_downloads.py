from ..darcy import DarcyDataset
from pathlib import Path

import os
import shutil

import pytest

test_data_dir = Path("./dataset_test")

@pytest.mark.parametrize('resolution', [16,32,64])
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
    
    shutil.rmtree(test_data_dir)