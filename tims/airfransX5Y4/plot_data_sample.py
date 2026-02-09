import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import json
import os

def plot_airfrans_sample(loader, index=0, device='cpu'):
    # Fetch batch
    for i, sample in enumerate(loader):
        if i == index // loader.batch_size:
            idx_in_batch = index % loader.batch_size
            x = sample['x'][idx_in_batch].to(device)
            y = sample['y'][idx_in_batch].to(device)
            break

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # --- INPUTS ---
    u_inf, v_inf, re_val = x[0,0,0].item(), x[1,0,0].item(), x[4,0,0].item()
    
    axes[0, 0].imshow(x[2].cpu(), origin='lower') 
    axes[0, 0].set_title(f"Mask (1.0 inside)\nU_inf: {u_inf:.2f}")
    
    axes[0, 1].imshow(x[3].cpu(), origin='lower', cmap='seismic')
    axes[0, 1].set_title(f"SDF\nV_inf: {v_inf:.2f}")
    
    # We can use the remaining input slots for the global params as text or placeholders
    axes[0, 2].axis('off')
    axes[0, 2].text(0, 0.5, f"Reynolds Log: {re_val:.4f}\nIndex: {index}", fontsize=12)
    axes[0, 3].axis('off')

    # --- OUTPUTS with Shared Scaling for Velocity ---
    out_titles = ["U-Deficit", "V-Deficit", "Cp (Pressure)", "log(nut_ratio)"]
    
    # Calculate shared limits for U and V (Indices 0 and 1)
    # We take the global min/max of both channels combined
    vel_min = min(y[0].min(), y[1].min()).cpu().item()
    vel_max = max(y[0].max(), y[1].max()).cpu().item()

    for i in range(4):
        # Choose colormap
        cmap = 'viridis' if i < 2 else ('plasma' if i == 2 else 'magma')
        
        # Apply shared scaling only for the first two plots
        if i < 2:
            im = axes[1, i].imshow(y[i].cpu(), origin='lower', cmap=cmap, vmin=vel_min, vmax=vel_max)
            axes[1, i].set_title(f"{out_titles[i]}\n(Shared Scale)")
        else:
            im = axes[1, i].imshow(y[i].cpu(), origin='lower', cmap=cmap)
            axes[1, i].set_title(out_titles[i])
            
        fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

    plt.suptitle(f"AirFrans X5Y4 | Shared Velocity Scaling: [{vel_min:.2f}, {vel_max:.2f}]", fontsize=16)
    plt.show()

def plot_airfrans_archive(archive_dir, split_name='full_train',grid_size=(128,128), index=0,device=0):

    manifest_path = Path(archive_dir) / "manifest.json"
    with open(manifest_path, 'r') as f:
        split_dict = json.load(f)
    
    filename = split_dict[split_name][index]
    print(" Get ", filename)
    full_filename = f"{filename}_archive_G{grid_size[0]}x{grid_size[1]}.pt"
    file_path = Path(archive_dir) / full_filename

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # Load with weights_only=False to handle the 'props' dictionary and any numpy scalars
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    
    x = data['x']             # [5, H, W]
    y_raw = data['y_raw']       # [4, H, W]
    y_ndim = data['y_ndim']  # [4, H, W]
    props = data['props']     # Metadata dictionary

    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # --- ROW 1: INPUTS ---
    # Access metadata directly from the props dict
    u_inf = props.get('v_inf', x[0,0,0].item())   # here v_inf should be v_mag_inf
    aoa = props.get('aoa_deg', 'N/A')
    re = props.get('reynolds', 'N/A')
    bounds = props.get('grid_bounds', [0,0,0,0])

    axes[0, 0].imshow(x[2], origin='lower') 
    axes[0, 0].set_title(f"Mask (Interior=1)\nU_inf: {u_inf:.2f}")

    axes[0, 1].imshow(x[3], origin='lower', cmap='seismic')
    axes[0, 1].set_title(f"SDF (Distance Map)\nAoA: {aoa}°")

    axes[0, 2].axis('off')
    metadata_text = (
        f"File: {os.path.basename(file_path)}\n\n"
        f"Reynolds: {re:.2e}\n"
        f"Bounds: {bounds}\n"
    )
    axes[0, 2].text(0, 0.5, metadata_text, fontsize=10, verticalalignment='center')
    axes[0, 3].axis('off')

    # --- ROW 2: OUTPUTS (Shared Velocity Scaling) ---
    v_min, v_max = y_raw[0:2].min().item(), y_raw[0:2].max().item()
    enc_v_min, enc_v_max = y_ndim[0:2].min().item(), y_ndim[0:2].max().item()
    
    titles_raw = ["U", "V", "Pressure", "nut"]
    titles_ndim = ["u_def", "v_def", "Cp (Pressure)", "log10(nut_ratio)"]

    cmaps = ['viridis', 'viridis', 'plasma', 'magma']

    n_levels = 10

    for i in range(4):
        v_min, v_max = y_raw[i].min().item(), y_raw[i].max().item()
        enc_v_min, enc_v_max = y_ndim[i].min().item(), y_ndim[i].max().item()     
        # Use shared scaling for velocity channels 0 and 1
        v_limits = {'vmin': v_min, 'vmax': v_max} if i < 2 else {}
        enc_v_limits = {'vmin': enc_v_min, 'vmax': enc_v_max} if i < 2 else {}
        
        levels_raw = np.linspace(v_min, v_max, n_levels)
        levels_raw = np.sort(np.unique(np.round(levels_raw, decimals=1)))
        if len(levels_raw) < 2:
            levels_raw = np.linspace(v_min, v_max, n_levels)

        im = axes[1, i].contourf(y_raw[i], levels=levels_raw, cmap=cmaps[i])

        axes[1, i].set_title(f"{titles_raw[i]} ")
        fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

        levels_ndim = np.linspace(enc_v_min, enc_v_max, n_levels)
        levels_ndim = np.sort(np.unique(np.round(levels_ndim, decimals=2)))
        if len(levels_ndim) < 2:
            levels_ndim = np.linspace(enc_v_min, enc_v_max, n_levels)

        im2 = axes[2, i].contourf(y_ndim[i], levels=levels_ndim, cmap=cmaps[i])
        axes[2, i].set_title(f"{titles_ndim[i]} ")
        fig.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)
    plt.suptitle(f"Archive Inspection: {split_name}  {index}   Grid {grid_size[0]}x{grid_size[1]}", fontsize=16)
    plt.show()


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tims.airfransX5Y4.airfrans_datasetX5Y4_v1 import AirfransDataset
    from tims.airfransX5Y4.config_AirfransX5Y4_v1 import AirfransDatasetConfig,Default
    from tims.airfransX5Y4.airfrans_datasetX5Y4_v1 import load_airfrans_dataset
    from zencfg import make_config_from_cli


    archive_dir = "/home/timm/Projects/PIML/Dataset_PT_FNO/Archive"    
    index = 8

    plot_airfrans_archive(archive_dir, split_name='aoa_train',grid_size=(128,128), index=index,device=0)

    # Example usage
    config = make_config_from_cli(Default)
    config = config.to_dict()

    train_loader, test_loaders, data_processor = load_airfrans_dataset(
            data_dir=config.data.data_dir,
            dataset_name=config.data.dataset_name,
            train_split=config.data.train_split,
            test_splits=config.data.test_splits,
            batch_size=config.data.batch_size,
            test_batch_sizes=config.data.test_batch_sizes,
            test_resolutions=config.data.test_resolutions,
            encode_input=config.data.encode_input,    
            encode_output=config.data.encode_output, 
            encoding=config.data.encoding,
            channel_dim=1,
        )

    plot_airfrans_sample(train_loader, index=index, device='cpu') 

