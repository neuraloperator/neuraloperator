from ast import Try
import os
import torch
import numpy as np
from scipy.interpolate import griddata
import airfrans as af
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import pyvista as pv
import concurrent.futures
from functools import partial


import pyvista as pv
import numpy as np



def process_airfrans_to_pt_archive(dataset_root, input_folder,archive_dir, training_dir ,xlen, ylen, xoffset, grid_size=(128, 128)):
    name = Path(input_folder).name



    simulation = af.Simulation(root=dataset_root, name=name)

    # 1. Extract Metadata from Name/Sim
    parts = name.split('_')
    v_mag_inf = float(parts[2]) 
    aoa_deg = float(parts[3])
    aoa_rad = np.deg2rad(aoa_deg)
    
    # Inlet Components
    u_inf = v_mag_inf * np.cos(aoa_rad)
    v_inf = v_mag_inf * np.sin(aoa_rad)
    nu_mol = simulation.NU
    rho = simulation.RHO
    reynolds = (v_mag_inf * 1.0 / nu_mol)  # assuming chord=1.0
    log_re = np.log10(reynolds)
    #print(f"Processing {name}: v_inf={v_inf}, aoa={aoa_deg}, nu_mol={nu_mol}, rho={rho}, log(Re)={log_re}")
    # 2. Grid Sampling
    mesh = simulation.internal
    xmin, xmax = (-xlen/2 + xoffset, xlen/2 + xoffset)
    ymin, ymax = (-ylen/2, ylen/2)

    x_range = np.linspace(xmin, xmax, grid_size[0])
    y_range = np.linspace(ymin, ymax, grid_size[1])
    grid = pv.RectilinearGrid(x_range, y_range, np.array([mesh.center[2]]))
    sampled = grid.sample(mesh)

    # 3. Raw Field Extraction
    sdf_raw = sampled.point_data['implicit_distance'].reshape(grid_size)
    u_raw = sampled.point_data['U'][:, 0].reshape(grid_size)
    v_raw = sampled.point_data['U'][:, 1].reshape(grid_size)
    p_raw = sampled.point_data['p'].reshape(grid_size)
    nut_raw = sampled.point_data['nut'].reshape(grid_size)

    # Clean NaNs immediately
    u_raw = np.nan_to_num(u_raw, nan=u_inf)
    v_raw = np.nan_to_num(v_raw, nan=v_inf)
    p_raw = np.nan_to_num(p_raw, nan=0.0)
    nut_raw = np.nan_to_num(nut_raw, nan=0.0)

    mask = np.where(sdf_raw < 0, 0.0, 1.0)

    # 4. Non-Dimensionalization (The "Step 3" Logic)
    # Velocity Deficit
    u_ndef = (u_inf - u_raw) / v_mag_inf   
    v_ndef = (v_inf - v_raw) / v_mag_inf   



    # set velocity deficit to 1.0 inside the body (sdf < 0)
    u_ndef[mask == 1.0] = 1.0
    v_ndef[mask == 1.0] = 1.0
    
    # Pressure Coefficient (Cp)
    cp = p_raw / (0.5 * rho * v_mag_inf**2)
    
    # Log Turbulent Viscosity Ratio (log10(nut/nu_mol))
    # We use a floor of 1e-10 to prevent log(0)
    log_nut_ratio = np.log10(np.clip((nut_raw + 1e-10) / nu_mol, a_min=1e-10, a_max=None))

    # check nu?_mol > 0
    if nu_mol <= 0:
        raise ValueError(f"Invalid molecular viscosity (nu_mol): {nu_mol}. Must be positive.")
    
    log_nut_ratio = np.log10(np.clip(nut_raw, 1e-12, None) / nu_mol)
    # 5. Build Inputs (x)

    sdf_fixed = np.nan_to_num(sdf_raw, nan=0.0)
    # x: [u_bc, v_bc, mask, sdf]
    x_data = np.stack([np.full_like(sdf_raw, u_inf), 
                       np.full_like(sdf_raw, v_inf), 
                       mask, sdf_fixed])

    # 6. Build Targets (y)
    # y_raw: Standard SI units
    y_raw = np.stack([u_raw, v_raw, p_raw, nut_raw])
    # y_ndim: Non-Dimensional units
    y_ndim = np.stack([u_ndef, v_ndef, cp, log_nut_ratio])


    airfoil_coords = Path(input_folder) / f"{name}_aerofoil.vtp"

    # For extracting the airfoil coordinates
    #foil = pv.read(airfoil_coords)
    foil = simulation.airfoil
    # Extract point coordinates
    foil_points = foil.points  # (N, 3) array

    #print(f" Airfoil points shape: {foil_points.shape}")

    # Calculate aerodynamic coefficients for verification
    ((cd, cdp, cdv), (cl, clp, clv)) = simulation.force_coefficient(compressible=False,reference=False)

    #print(f"  Cd: {cd:.5f}, Cdp: {cdp:.5f}, Cdv: {cdv:.5f}, Cl: {cl:.5f}, Clp: {clp:.5f}, Clv: {clv:.5f}")

    # 7. Package everything
    archive_dict = {
        'x': torch.tensor(x_data, dtype=torch.float32),
        'y_raw': torch.tensor(y_raw, dtype=torch.float32),
        'y_ndim': torch.tensor(y_ndim, dtype=torch.float32),
        'props': {
            'v_mag_inf': torch.tensor(v_mag_inf, dtype=torch.float32),
            'aoa_deg':  torch.tensor(aoa_deg, dtype=torch.float32),
            'nu_mol':  torch.tensor(nu_mol, dtype=torch.float32),
            'rho':  torch.tensor(rho, dtype=torch.float32),
            'reynolds': torch.tensor(reynolds, dtype=torch.float32),
            'log_reynolds': torch.tensor(log_re, dtype=torch.float32),
            'grid_bounds': torch.tensor([xmin, xmax, ymin, ymax], dtype=torch.float32),
            'airfoil_points': torch.from_numpy(foil_points).to(torch.float32),
            'cd': torch.tensor(cd, dtype=torch.float32),
            'cl': torch.tensor(cl, dtype=torch.float32),
            'cdp': torch.tensor(cdp, dtype=torch.float32),
            'clp': torch.tensor(clp, dtype=torch.float32),
            'cdv': torch.tensor(cdv, dtype=torch.float32),
            'clv': torch.tensor(clv, dtype=torch.float32)
        }
    }

    # Non-dim BCs
    u_bc = np.cos(aoa_rad)
    v_bc = np.sin(aoa_rad)

    # Package into 5-channel input for normalized training
    x_train = np.stack([
        np.full_like(sdf_raw, u_bc), 
        np.full_like(sdf_raw, v_bc),     
        mask, 
        sdf_fixed,
        np.full_like(sdf_raw, log_re), 
    ])



    # 8. Package specifically for the Training Loop (Non-Dimensional + props)
    training_dict = {
        'x': torch.tensor(x_train, dtype=torch.float32),
        'y': torch.tensor(y_ndim, dtype=torch.float32),
        'props': {
            'v_mag_inf': torch.tensor(v_mag_inf, dtype=torch.float32),
            'aoa_deg':  torch.tensor(aoa_deg, dtype=torch.float32),
            'nu_mol':  torch.tensor(nu_mol, dtype=torch.float32),
            'rho':  torch.tensor(rho, dtype=torch.float32),
            'reynolds': torch.tensor(reynolds, dtype=torch.float32),
            'log_reynolds': torch.tensor(log_re, dtype=torch.float32),
            'grid_bounds': torch.tensor([xmin, xmax, ymin, ymax], dtype=torch.float32),
            'airfoil_points': torch.from_numpy(foil_points).to(torch.float32),
            'cd': torch.tensor(cd, dtype=torch.float32),
            'cl': torch.tensor(cl, dtype=torch.float32),
            'cdp': torch.tensor(cdp, dtype=torch.float32),
            'clp': torch.tensor(clp, dtype=torch.float32),
            'cdv': torch.tensor(cdv, dtype=torch.float32),
            'clv': torch.tensor(clv, dtype=torch.float32)
        }
    }

    # 9. Save both to distinct locations
    archive_path = Path(archive_dir) / f"{name}_archive_G{grid_size[0]}x{grid_size[1]}.pt"
    train_path = Path(training_dir) / f"{name}_train_G{grid_size[0]}x{grid_size[1]}.pt"

    torch.save(archive_dict, archive_path)
    torch.save(training_dict, train_path)


# 1. Move the wrapper OUTSIDE any other function
def global_worker_wrapper(arg_tuple):
    """
    Takes a single tuple of arguments because executor.map 
    is happiest with a single iterable.
    """
    sim_dir, dataset_root, archive_dir, training_dir, xlen, ylen, xoffset, grid_size = arg_tuple
    
    return process_airfrans_to_pt_archive(
        input_folder=sim_dir,
        dataset_root=dataset_root,
        archive_dir=archive_dir,
        training_dir=training_dir,
        xlen=xlen,
        ylen=ylen,
        xoffset=xoffset,
        grid_size=grid_size
    )


def convert(dataset_root, output_folder, xlen, ylen, xoffset, grid_size):
    archive_dir = Path(output_folder) / "Archive"
    training_dir = Path(output_folder) / "TrainingX5Y4"
    Path(archive_dir).mkdir(parents=True, exist_ok=True)
    Path(training_dir).mkdir(parents=True, exist_ok=True)

    sim_dirs = [str(d) for d in Path(dataset_root).iterdir() if d.is_dir()]
    
# 2. Prepare the list of argument tuples for the map
    # This bundles everything the worker needs into one package per simulation
    job_args = [
        (d, dataset_root, archive_dir, training_dir, xlen, ylen, xoffset, grid_size) 
        for d in sim_dirs
    ]
    
    # If num_workers is None, it uses all available logical cores - 2.

    num_workers=16
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 2)


    
    print(f"🚀 Starting Multiprocessing with {num_workers} workers...")

    # We use ProcessPoolExecutor for CPU-bound tasks like PyVista sampling
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map the sim_dirs to the processing function
        list(tqdm(executor.map(global_worker_wrapper, job_args), total=len(job_args), desc="Processing Airfoils"))


def get_dataset_stats(training_dir):
    all_x = []
    all_y = []
    files = list(Path(training_dir).glob("*.pt"))
    
    for f in tqdm(files[:100], desc="Sampling stats"): # Sample 100 for speed
        data = torch.load(f)
        all_x.append(data['x'])
        all_y.append(data['y'])
    
    x_stack = torch.stack(all_x)
    y_stack = torch.stack(all_y)
    
    # Calculate across [Batch, Height, Width] but keep Channels
    x_mean = x_stack.mean(dim=(0, 2, 3))
    x_std = x_stack.std(dim=(0, 2, 3))
    y_mean = y_stack.mean(dim=(0, 2, 3))
    y_std = y_stack.std(dim=(0, 2, 3))
    
    return x_mean, x_std, y_mean, y_std

if __name__ == "__main__":

    PATH_TO_DATASET = "/home/timm/Projects/PIML/Dataset"
    PATH_TO_OUTPUT = "/home/timm/Projects/PIML/Dataset_PT_FNO_X5Y4"
    GRID_SIZE = (64, 64)
    REGENERATE = True  # Set to True to regenerate all pt files
    xlen = float(6.0)   # domain length to be sampled  (-xlen/2, xlen/2)
    ylen = float(3.0)   # domain height to be sampled  (-ylen/2, ylen/2)
    xoffset = float(1.0)  # x-offset to be sampled
    # Rectangular grids
    #convert(PATH_TO_DATASET,PATH_TO_OUTPUT, xlen,ylen,xoffset, GRID_SIZE )
    #GRID_SIZE = (128, 128)
    #convert(PATH_TO_DATASET,PATH_TO_OUTPUT, xlen,ylen,xoffset, GRID_SIZE )
    #GRID_SIZE = (256, 256)
    #convert(PATH_TO_DATASET,PATH_TO_OUTPUT, xlen,ylen,xoffset, GRID_SIZE )
    #GRID_SIZE = (512, 512)
    #convert(PATH_TO_DATASET,PATH_TO_OUTPUT, xlen,ylen,xoffset, GRID_SIZE )
    GRID_SIZE = (1024, 1024)
    convert(PATH_TO_DATASET,PATH_TO_OUTPUT, xlen,ylen,xoffset, GRID_SIZE )   

    manifest_path = Path(PATH_TO_DATASET) / "manifest.json"
    #copy to output directories
    import shutil
    shutil.copy(manifest_path, Path(PATH_TO_OUTPUT) / "Archive" / "manifest.json")
    shutil.copy(manifest_path, Path(PATH_TO_OUTPUT) / "TrainingX5Y4" / "manifest.json")