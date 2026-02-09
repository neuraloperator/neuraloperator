import torch
import matplotlib.pyplot as plt
import os

def plot_airfrans_archive(file_path, plot_type='y_raw'):
    """
    Loads a .pt archive file with specific keys: 'x', 'y_raw', 'y_ndim', 'props'
    plot_type: 'y_raw' for physical units, 'y_ndim' for non-dimensionalized units
    """
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
    u_inf = props.get('v_inf', x[0,0,0].item())
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
        f"Plotting: {plot_type}"
    )
    axes[0, 2].text(0, 0.5, metadata_text, fontsize=10, verticalalignment='center')
    axes[0, 3].axis('off')

    # --- ROW 2: OUTPUTS (Shared Velocity Scaling) ---
    v_min, v_max = y_raw[0:2].min().item(), y_raw[0:2].max().item()
    enc_v_min, enc_v_max = y_ndim[0:2].min().item(), y_ndim[0:2].max().item()
    
    titles_raw = ["u_def", "v_def", "Cp (Pressure)", "log10(nut_ratio)"]
    titles_ndim = ["u_def", "v_def", "Cp (Pressure)", "log10(nut_ratio)"]

    cmaps = ['viridis', 'viridis', 'plasma', 'magma']

    for i in range(4):
        # Use shared scaling for velocity channels 0 and 1
        v_limits = {'vmin': v_min, 'vmax': v_max} if i < 2 else {}
        enc_v_limits = {'vmin': enc_v_min, 'vmax': enc_v_max} if i < 2 else {}
        
        im = axes[1, i].imshow(y_raw[i], origin='lower', cmap=cmaps[i], **v_limits)
        axes[1, i].set_title(f"{titles_raw[i]} ")
        fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        im2 = axes[2, i].imshow(y_ndim[i], origin='lower', cmap=cmaps[i], **enc_v_limits)
        axes[2, i].set_title(f"{titles_ndim[i]} ")
        fig.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)
    plt.suptitle(f"Archive Inspection: {plot_type.upper()} Mode", fontsize=16)
    plt.show()

# Usage:
path = "/home/timm/Projects/PIML/Dataset_PT_FNO/Archive/airFoil2D_SST_31.68_0.424_0.273_4.301_1.0_11.616_archive_G128x128.pt"
plot_airfrans_archive(path, plot_type='y_raw', )