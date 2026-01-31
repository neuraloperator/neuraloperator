import torch
import numpy as np
import matplotlib.pyplot as plt
from cartesianAirfransDataset import CartesianAirfransDataset
from cartesianAirfransDataset import AirfoilNormalizer
import json
from pathlib import Path

def verify_normalization_plots(dataset, idx=0):
    # 1. Get raw data from the file directly

    sim_dir_path =  dataset.sim_dirs[idx]
    print(f" Path  {sim_dir_path}")
    airfoil_name = sim_dir_path.name
    print(f" airfoil {airfoil_name}")
    data_file = Path(sim_dir_path) / f"{airfoil_name}_X{dataset.xlim}_Y{dataset.ylim}_G{dataset.grid_size[0]}x{dataset.grid_size[1]}.pt"

    print( f" Loading {data_file}")
    raw_data = torch.load(data_file, weights_only=True)

    sdf_idx = 3
    press_idx = 6
    nut_idx = 7
    u_target = 4
    v_target = 5

    raw_p = raw_data[press_idx] # Raw pressure
    raw_nut = raw_data[nut_idx] # Raw nut
    raw_sdf = raw_data[sdf_idx] # Raw nut
    
    # 2. Get normalized data from the Dataset (via transform)
    x_norm, y_norm = dataset[idx]
    norm_p = y_norm[press_idx-4] # Normalized pressure 
    norm_nut = y_norm[nut_idx-4] # Normalized nut 
    norm_sdf = x_norm[sdf_idx]  # normalized sdf
    # 3. Plotting
    fig, axes = plt.subplots(3, 4, figsize=(8,12))
    
    # Raw Pressure Plot
    im0 = axes[0,0].imshow(raw_p, origin='lower', cmap='RdBu_r')
    axes[0,0].set_title(f"Raw Pressure (Pa)\nMax: {raw_p.max():.2f}")
    fig.colorbar(im0, ax=axes[0,0])
    
    # Normalized Pressure Plot
    im1 = axes[0,1].imshow(norm_p, origin='lower', cmap='RdBu_r')
    axes[0,1].set_title(f"Normalized Pressure (Z-score)\nMax: {norm_p.max():.2f}")
    fig.colorbar(im1, ax=axes[0,1])

    # Raw Nut Plot
    im2 = axes[1,0].imshow(raw_nut, origin='lower', cmap='RdBu_r')
    axes[1,0].set_title(f"Raw Nut (Pa)\nMax: {raw_nut.max():.2f}")
    fig.colorbar(im2, ax=axes[1,0])
    
    # Normalized Nut Plot
    im3 = axes[1,1].imshow(norm_nut, origin='lower', cmap='RdBu_r')
    axes[1,1].set_title(f"Normalized Nut (Z-score)\nMax: {norm_nut.max():.2f}")
    fig.colorbar(im3, ax=axes[1,1])

    # Raw SDF Plot
    im4 = axes[2,0].imshow(raw_sdf, origin='lower', cmap='RdBu_r')
    axes[2,0].set_title(f"Raw SDF (Pa)\nMin: {raw_sdf.min():.2f}")
    fig.colorbar(im4, ax=axes[2,0])
    
    # Normalized SDF Plot
    im5 = axes[2,1].imshow(norm_sdf, origin='lower', cmap='RdBu_r')
    axes[2,1].set_title(f"Normalized SDF (Z-score)\nMin: {norm_sdf.min():.2f}")
    fig.colorbar(im5, ax=axes[2,1])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":


    PATH_TO_DATASET = "/home/timm/Projects/PIML/Dataset"
    GRID_SIZE = (128, 128)
    REGENERATE = True  # Set to True to regenerate all pt files
    xlen = 2   # domain length to be sampled  (-xlen/2, xlen/2)
    ylen = 2   # domain height to be sampled  (-ylen/2, ylen/2)


    # load dataset
    # Load Stats JSON and convert back to tensors
    stats_file = Path(PATH_TO_DATASET) / "statistics.json"
    with open(stats_file, 'r') as f:
        stats_dict = json.load(f)

    print(f"means  {stats_dict['means']}")
    input_indices = [0,1,3]
    target_indices=[4,5,6,7]
    for ch in target_indices:
        print(f" mean {stats_dict['means'][ch]}   std {stats_dict['stds'][ch]} ")

    normalizer = AirfoilNormalizer(stats_dict,input_indices,target_indices)
    dataset = CartesianAirfransDataset(PATH_TO_DATASET, split = 'all',xlim=2,ylim=2,  grid_size=(128,128), input_channels=[0,1,2,3],target_channels=[4,5,6,7] ,transform=normalizer )

    verify_normalization_plots(dataset, idx=0)
