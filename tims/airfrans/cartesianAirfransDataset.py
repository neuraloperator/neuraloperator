"""
Dataset classes for airfrans geometries and flow fields.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from cartesianAirfoil2DDataset import CartesianAirfoil2DDataset
import json
from torch.utils.data import DataLoader


import torch

class AirfoilNormalizer:
    def __init__(self, stats,input_indices = [0,1,3], target_indices=[4,5,6,7]):
        """
        stats: Dictionary containing 'means', 'stds', 'maxs', 'mins' as tensors
        target_indices: The indices of the target channels within the stats list
        """
        self.means = stats['means']
        self.stds = stats['stds']
        self.maxs = stats['maxs']
        self.input_indices = input_indices
        self.target_indices = target_indices


    def __call__(self, x, y):

        for i, data_idx in enumerate(self.input_indices):
            # i is the local index in the x slice
            # global_idx is the index in the stats file
            x[i] = (x[i] - self.means[data_idx]) / (self.stds[data_idx] + 1e-8)

        for i, data_idx in enumerate(self.target_indices):
            mean = self.means[data_idx]
            std = self.stds[data_idx]
            y[i] = (y[i] - mean) / (std + 1e-8)
            
        
        return x, y

class CartesianAirfransDataset(CartesianAirfoil2DDataset):

    def __init__(self, data_root, split='full_train', xlim=2, ylim=2,  grid_size=(128, 128), input_channels=[0,1,2,3],target_channels=[4,5,6] ,transform=None):
        """
        Initialize AirFRANS dataset.
        
        Args:
            data_root: Root directory of AirFRANS dataset
            split: 'train', 'val', 'full_train', 'scarce_train', 'aoa_train', 'reynolds_train' or 'test'
            grid_size: Target grid size
            input_channels: [u_input, v_input, mask, sdf]
            output_channels: [u_target, v_target, p_target]
            transform: Optional transform
        """
        self.data_root = Path(data_root)
        self.split = split
        self.grid_size = grid_size
        self.xlim =xlim
        self.ylim = ylim
        self.input_channels =input_channels
        self.output_channels =target_channels
        self.transform = transform


        
        # Find all simulation directories
        self.sim_dirs = sorted([d for d in Path(self.data_root).iterdir() if d.is_dir()])

        # Airfrans is delivered with a manifest defining training and test sets
        self.manifest = Path(data_root) / "manifest.json"


        # Split the data
        total = len(self.sim_dirs)
        if split == 'train':
            self.sim_dirs = self.sim_dirs[:int(0.8 * total)]
        elif split == 'val':
            self.sim_dirs = self.sim_dirs[int(0.8 * total):int(0.9 * total)]
        elif split =='all':
            self.sim_dirs = self.sim_dirs
        elif split == 'full_train':
            try :
                with open(self.manifest, 'r') as f:

                    manifest_data = json.load(f)
                    train_dirs = manifest_data.get('full_train', [])
                    self.sim_dirs = [self.data_root / d for d in train_dirs if (self.data_root / d).is_dir()]
            except FileNotFoundError:
                print(f"Manifest file not found at {self.manifest}. Using default split.")
                self.sim_dirs = self.sim_dirs[:int(0.8 * total)]    
        elif split == 'full_test':
            try :
                with open(self.manifest, 'r') as f:
                    manifest_data = json.load(f)
                    train_dirs = manifest_data.get('full_test', [])
                    self.sim_dirs = [self.data_root / d for d in train_dirs if (self.data_root / d).is_dir()]
            except FileNotFoundError:
                print(f"Manifest file not found at {self.manifest}. Using default split.")
                self.sim_dirs = self.sim_dirs[:int(0.2 * total)]                   
        elif split == 'scarce_train':
            try :
                with open(self.manifest, 'r') as f:
                    manifest_data = json.load(f)
                    train_dirs = manifest_data.get('scarce_train', [])
                    self.sim_dirs = [self.data_root / d for d in train_dirs if (self.data_root / d).is_dir()]
            except FileNotFoundError:
                print(f"Manifest file not found at {self.manifest}. Using 0.2 split.")
                self.sim_dirs = self.sim_dirs[:int(0.2 * total)]               
        elif split == 'aoa_train':
            try :
                with open(self.manifest, 'r') as f:
                    manifest_data = json.load(f)
                    train_dirs = manifest_data.get('aoa_train', [])
                    self.sim_dirs = [self.data_root / d for d in train_dirs if (self.data_root / d).is_dir()]
            except FileNotFoundError:
                print(f"Manifest file not found at {self.manifest}. Using 0.5 split.")
                self.sim_dirs = self.sim_dirs[:int(0.5 * total)]
        elif split == 'aoa_test':
            try :
                with open(self.manifest, 'r') as f:
                    manifest_data = json.load(f)
                    val_dirs = manifest_data.get('aoa_test', [])
                    self.sim_dirs = [self.data_root / d for d in val_dirs if (self.data_root / d).is_dir()]
            except FileNotFoundError:
                print(f"Manifest file not found at {self.manifest}. Using 0.2 split.")
                self.sim_dirs = self.sim_dirs[int(0.2 * total):int(0.4 * total)]
        else:  # test
            self.sim_dirs = self.sim_dirs[int(0.9 * total):]
    
    def __len__(self):
        return len(self.sim_dirs)
    
    def get_statistics(self):
        return self.stats

        
    def __getitem__(self, idx):
        """Load AirFRANS simulation data."""
        sim_dir = self.sim_dirs[idx]
        
        data_file = Path(sim_dir) / f"{sim_dir.name}_X{self.xlim}_Y{self.ylim}_G{self.grid_size[0]}x{self.grid_size[1]}.pt"
        # AirFRANS typically has multiple NPZ files per simulation
        # Adapt this based on actual AirFRANS structure
        if data_file.exists():
            data = torch.load(data_file)
            # Slice the master tensor based on your experiment needs
            inputs = data[self.input_channels, :, :]  # e.g., [Mask, SDF]
            targets = data[self.output_channels, :, :] # e.g., [u,v,p]
            
            
            if self.transform:
                inputs,outputs = self.transform(inputs,targets)
            
            return inputs,outputs

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data_files)

    def _extract_conditions(self, sim_dir, data):
        """Extract flow conditions from directory name or data."""
        # Example: parse from directory name like "airFoil2D_SST_31.803_7.291_..."
        dir_name = sim_dir.name
        
        if 'alpha' in data:
            alpha = data['alpha']
        else:
            # Parse from directory name
            alpha = 0.0
        
        Re = data.get('Re', 1e6)
        Mach = data.get('Mach', 0.3)
        
        return np.array([alpha, Re, Mach])



def load_airfrans(
    n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    data_root=example_data_root,
    test_resolutions=[16, 32],
    encode_input=False,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,
):
    dataset = CartesianAirfransDataset(
        root_dir=data_root,
        n_train=n_train,
        n_tests=n_tests,
        batch_size=batch_size,
        test_batch_sizes=test_batch_sizes,
        train_resolution=16,
        test_resolutions=test_resolutions,
        encode_input=encode_input,
        encode_output=encode_output,
        channel_dim=channel_dim,
        encoding=encoding,
        download=True,
    )

    # return dataloaders for backwards compat
    train_loader = DataLoader(
        dataset.train_db,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
    )

    test_loaders = {}
    for res, test_bsize in zip(test_resolutions, test_batch_sizes):
        test_loaders[res] = DataLoader(
            dataset.test_dbs[res],
            batch_size=test_bsize,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            persistent_workers=False,
        )

    return train_loader, test_loaders, dataset.data_processor