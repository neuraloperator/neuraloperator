import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
from pathlib import Path

def write_statistics(root_dir, xlen,ylen, grid_size=(128,128)):

            # Find all simulation directories
    sim_dirs = sorted([d for d in Path(root_dir).iterdir() if d.is_dir()])

    # Detect channel count from the first file
    sample_file = Path(sim_dirs[0]) / f"{sim_dirs[0].name}_X{xlen}_Y{ylen}_G{grid_size[0]}x{grid_size[1]}.pt"

    sample_data = torch.load(sample_file, weights_only=True)
    num_channels = sample_data.shape[0] # Correctly access C from [C, H, W]

    sum_vals = torch.zeros(num_channels)
    sq_sum_vals = torch.zeros(num_channels)
    num_elements = 0
            
    # Initialize mins to infinity and maxs to negative infinity
    mins = torch.full((num_channels,), float('inf'))
    maxs = torch.full((num_channels,), float('-inf'))
    
    num_elements = 0
    for sim_dir in sim_dirs:
        data_file = Path(sim_dir) / f"{sim_dir.name}_X{xlen}_Y{ylen}_G{grid_size[0]}x{grid_size[1]}.pt"

        data = torch.load(data_file, weights_only=True) # Shape: [C, H, W]
        for i in range(num_channels):
            channel_data = data[i]
            
            # Statistics for Z-score
            sum_vals[i] += channel_data.sum()
            sq_sum_vals[i] += (channel_data**2).sum()
            
            # Statistics for Min-Max 
            mins[i] = torch.min(mins[i], channel_data.min())
            maxs[i] = torch.max(maxs[i], channel_data.max())
            
        num_elements += data[0].numel()
           
    
    # Final calculations
    means = sum_vals / num_elements
    stds = torch.sqrt((sq_sum_vals / num_elements) - (means**2))

    stats = {
        'means': means.tolist(),
        'stds': stds.tolist(),
        'mins': mins.tolist(),
        'maxs': maxs.tolist(),
        'num_channels': num_channels
    }

    # --- Printout Section ---
    channel_names = ["U_inf", "V_inf",  "Mask", "SDF",  "U_target", "V_target", "P_target","Nut"]
    
    print("\n" + "="*80)
    print(f"{'Channel':<12} | {'Mean':>12} | {'Std':>12} | {'Min':>12} | {'Max':>12}")
    print("-" * 80)
    
    for i in range(num_channels):
        name = channel_names[i] if i < len(channel_names) else f"Ch_{i}"
        print(f"{name:<12} | {means[i]:12.4e} | {stds[i]:12.4e} | {mins[i]:12.4e} | {maxs[i]:12.4e}")
    print("="*80 + "\n")

    # Save to JSON
    with open(Path(root_dir) / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=4)



if __name__ == "__main__":

    PATH_TO_DATASET = "/home/timm/Projects/PIML/Dataset"
    GRID_SIZE = (256, 256)
    REGENERATE = True  # Set to True to regenerate all pt files
    xlen = 2   # domain length to be sampled  (-xlen/2, xlen/2)
    ylen = 2   # domain height to be sampled  (-ylen/2, ylen/2)
    stats = write_statistics(PATH_TO_DATASET, xlen,ylen, GRID_SIZE)

