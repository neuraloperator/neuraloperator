import torch
import numpy as np
from scipy.interpolate import griddata
import airfrans as af
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import pyvista as pv

def process_airfrans_to_fno(dataset_root, input_folder, xlen ,ylen , grid_size=(128, 128)):
    name = Path(input_folder).name # Get airfoil name from folder path

    simulation = af.Simulation(root = dataset_root, name = name)


    parts = name.split('_')
    v_inf = float(parts[2]) 
    aoa = float(parts[3])  
    naca = parts[4:]
    nacatype = 'NACA' + '_'.join(naca)


    print("Simulation loaded:")
    print(f" NACA 4-5 info :  {nacatype}" )
    print(f"Velocity: {v_inf} m/s")
    print(f"AoA: {aoa} deg")
    print(f"kinematic viscosity: {simulation.NU}")
    print(f"density: {simulation.RHO}")
    reynolds = (v_inf * 1 / simulation.NU)
    print(f"Reynolds number: {reynolds}")
    print()

    airfoil_coords = Path(input_folder) / f"{name}_aerofoil.vtp"
    #airfoil_freestream = Path(input_folder) / f"{airfoil_name}_freestream.vtp"
    airfoil_flowfield = Path(input_folder) / f"{name}_internal.vtu"
    
    # Read the VTP file
    foil = pv.read(airfoil_coords)
    
    # Extract point coordinates
    foil_points = foil.points  # (N, 3) array

    print(f" Airfoil points shape: {foil_points.shape}")
    # 

    # Load the internal mesh
    mesh = pv.read(airfoil_flowfield)
    
    # Define FNO Grid (Adjust bounds based on your AirFrans domain)
    # Clipped Domain Size [(-2, 4), (-1.5, 1.5), (0, 1)]
    # shift x by 0.5
    xmin = -xlen/2 + 0.5
    xmax = xlen/2 + 0.5
    ymin = -ylen/2
    ymax = ylen/2


    x_range = np.linspace(xmin, xmax, grid_size[1])
    y_range = np.linspace(ymin, ymax, grid_size[0])
    grid = pv.RectilinearGrid(x_range, y_range, np.array([mesh.center[2]]))

    # Sample and Reshape
    sampled = grid.sample(mesh)

    sdf_raw = sampled.point_data['implicit_distance'].reshape(grid_size)
    u_raw = sampled.point_data['U'][:, 0].reshape(grid_size)
    v_raw = sampled.point_data['U'][:, 1].reshape(grid_size)
    p_raw = sampled.point_data['p'].reshape(grid_size)
    cp_raw = p_raw / (0.5 * simulation.RHO * v_inf**2)
    nut_raw = sampled.point_data['nut'].reshape(grid_size)



    # 1. Create the binary mask (1 for fluid, 0 for airfoil)
    # Use SDF sign: negative SDF = inside airfoil = 0, positive SDF = fluid = 1
    mask_binary = np.where(sdf_raw < 0, 0.0, 1.0)

    # input boundary conditions to fields u_input v_input 
    aoa_rad = np.deg2rad(aoa)
    u_input = v_inf*np.ones_like(sdf_raw)*np.cos(aoa_rad)
    v_input = v_inf*np.ones_like(sdf_raw)*np.sin(aoa_rad)

    # 2. Save the SDF field separately (useful for visualization/analysis)
    sdf_fixed = np.nan_to_num(sdf_raw, nan=0.0)

    # 3. Force physics outputs (u, v, cp) to 0 inside the foil
    u_target = np.nan_to_num(u_raw, nan=0.0)
    v_target = np.nan_to_num(v_raw, nan=0.0)
    cp_target = np.nan_to_num(cp_raw, nan=0.0)
    nut_fixed = np.nan_to_num(nut_raw, nan=0.0)
    # Separate input and target data for PTDataset format
    x_data = np.stack([u_input, v_input, mask_binary, sdf_fixed])  # Input: BC + geometry
    y_data = np.stack([u_target, v_target, cp_target, nut_fixed])   # Target: flow fields
    
    # Convert to torch tensors
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    
    # Create data dictionary for PTDataset compatibility
    data_dict = {
        'x': x_tensor,
        'y': y_tensor
    }

    airfoil_name = Path(input_folder).name # Get airfoil name from folder path

    output_file = Path(input_folder) / f"{airfoil_name}_Cp_X{xlen}_Y{ylen}_G{grid_size[0]}x{grid_size[1]}.pt"

    # Save as dictionary for PTDataset compatibility
    torch.save(data_dict, output_file)
    print(f"Processed: {output_file}")


def convert(dataset_root,xlen, ylen, grid_size):
    

    sim_dirs = [str(d) for d in Path(dataset_root).iterdir() if d.is_dir()]
    pbar1 = tqdm(sim_dirs, desc="Extracting VTPs and convert pressure to Cp save as pt")
    # Proess each simulation directory to generate NPZ files
    for sim_dir in pbar1:
        
        pbar1.set_description(f"Converting  {sim_dir}  {sim_dirs.index(sim_dir)+1}/{len(sim_dirs)}")

        output_file = Path(sim_dir) / f"{Path(sim_dir).name}_Cp_X{xlen}_Y{ylen}_G{grid_size[0]}x{grid_size[1]}.pt"
        if REGENERATE or not output_file.exists():
            process_airfrans_to_fno(dataset_root, sim_dir, xlen, ylen,grid_size)
        else:
            print(f"Skipping {sim_dir} (pt file exists)")
            #process_airfrans_to_fno(dataset_root, sim_dir, xlen, ylen,grid_size)

if __name__ == "__main__":

    PATH_TO_DATASET = "/home/timm/Projects/PIML/Dataset"
    GRID_SIZE = (64, 64)
    REGENERATE = True  # Set to True to regenerate all pt files
    xlen = 2   # domain length to be sampled  (-xlen/2, xlen/2)
    ylen = 2   # domain height to be sampled  (-ylen/2, ylen/2)
    convert(PATH_TO_DATASET, xlen,ylen, GRID_SIZE)
    GRID_SIZE = (128, 128)
    convert(PATH_TO_DATASET,xlen,ylen,GRID_SIZE)
    GRID_SIZE = (256, 256)
    convert(PATH_TO_DATASET,xlen,ylen,GRID_SIZE)