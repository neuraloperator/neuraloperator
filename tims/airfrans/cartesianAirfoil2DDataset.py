"""
Dataset classes for airfoil geometries and flow fields.
"""

import torch
from .pt_dataset import PTDataset
import numpy as np
from pathlib import Path


class CartesianAirfoil2DDataset(PTDataset):
    """
    Dataset for airfoil geometries and corresponding flow fields.
    
    This dataset handles loading and preprocessing of:
    - Airfoil coordinates (x, y)
    - Flow field variables (u, v, p) on a grid
    - Flow conditions (angle of attack, Reynolds number, Mach number)
    """
    
    def __init__(self, root, split='train', transform=None, grid_size=(128, 128)):
        """
        Initialize airfoil dataset.
        
        Args:
            data_dir: Directory containing the dataset
            split: Dataset split ('train', 'val', or 'test')
            transform: Optional transform to apply to data
            grid_size: Size of the computational grid (height, width)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.grid_size = grid_size
        
        # Load dataset indices
        self.data_files = sorted(list((self.data_dir / split).glob("*.pt")))
        
        if len(self.data_files) == 0:
            print(f"Warning: No data files found in {self.data_dir / split}")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data_files)
    
    
    def get_statistics(self):
        """
        Compute dataset statistics for normalization.
        
        Returns:
            Dictionary with mean and std for flow field variables
        """
        all_u, all_v, all_p, all_alpha, all_reynolds = [], [], [],[], []
        
        for idx in range(len(self)):
            data = np.load(self.data_files[idx])
            all_u.append(data.get('u', np.zeros(self.grid_size)).flatten())
            all_v.append(data.get('v', np.zeros(self.grid_size)).flatten())
            all_p.append(data.get('p', np.zeros(self.grid_size)).flatten())
            all_alpha.append(data.get('alpha'))
            all_reynolds.append(data.get('Re'))
        
        all_u = np.concatenate(all_u)
        all_v = np.concatenate(all_v)
        all_p = np.concatenate(all_p)
        all_alpha = np.array(all_alpha)
        
        stats = {
            'u_mean': float(np.mean(all_u)),
            'u_std': float(np.std(all_u)),
            'v_mean': float(np.mean(all_v)),
            'v_std': float(np.std(all_v)),
            'p_mean': float(np.mean(all_p)),
            'p_std': float(np.std(all_p)),
            'alpha_mean': float(np.mean(all_alpha)),
            'alpha_std': float(np.std(all_alpha)),
            'alpha_min': float(np.min(all_alpha)),
            'alpha_max': float(np.max(all_alpha))   
        }
        
        return stats