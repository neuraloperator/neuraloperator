    
from pathlib import Path
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from torch.utils.data._utils.collate import default_collate
class AirfransUtils:

    def __init__(self, *, model, data_processor, device):
        self.model = model
        self.data_processor = data_processor
        self.device = device

    def collate_with_props(batch):
        """
        batch: a list of dicts from Dataset.__getitem__
        """
        # 1. Separate the props from the tensors
        props = [item.pop('props') for item in batch]
        
        # 2. Use the standard collate for x and y (which are same shape)
        # This creates the standard [B, C, H, W] tensors
        collated_batch = default_collate(batch)
    
        # 3. Put the props back as a raw list of dictionaries
        collated_batch['props'] = props
        
        return collated_batch
    
    @torch.no_grad()
    def evaluate_batch_metrics(self, batch):
        """Returns a list of predicted CL values for the entire batch"""
        y_pred = self.model(batch['x'].to(self.device))
        # Filter out only tensor items for postprocess
        tensor_batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        decoded_out, _ = self.data_processor.postprocess(y_pred, tensor_batch)  
        y_phys = decoded_out['y']
        
        props_batch = batch['props']
        cl_results = []

        batch_size = y_phys.size(0)

        print(f"DEBUG: props_batch type: {type(props_batch)}")
        if isinstance(props_batch, dict):
            print(f"DEBUG: props_batch keys: {props_batch.keys()}")

        for i in range(batch_size):
            # 1. Check if we need to reconstruct the individual dict
            if isinstance(props_batch, dict):
                # Pull the i-th element for every key in the metadata
                props = {k: v[i] for k, v in props_batch.items()}
            else:
                # If it's a list/tuple of dicts (standard)
                props = props_batch[i]

            # 2. Extract and integrate
            foil_coords = props['airfoil_points']
            if hasattr(foil_coords, 'to'):
                foil_coords = foil_coords.to(self.device)
            
            # Now props['grid_bounds'] and props['aoa_deg'] will be available
            clp = self.calculate_cl_from_cp_sdf(
                surface_cp=self.get_surface_cp_on_foil(y_phys[i:i+1], foil_coords, props['grid_bounds']),
                foil_coords=foil_coords,
                sdf_grid=batch['x'][i:i+1, 3:4], 
                grid_bounds=props['grid_bounds'],
                aoa_deg=props['aoa_deg']
            )
            cl_results.append(clp)
            
        return cl_results

    def get_surface_cp_on_foil(self, y_pred, foil_coords, grid_bounds):
        """
        y_pred: [1, 4, H, W] tensor (u, v, cp, nut)
        foil_coords: [N, 2] physical coordinates of the foil
        grid_bounds: [xmin, xmax, ymin, ymax]
        """
        # 1. Normalize foil_coords to [-1, 1] for grid_sample
        xmin, xmax, ymin, ymax = grid_bounds
        x_norm = 2.0* (foil_coords[:, 0] - xmin) / (xmax - xmin) - 1.0
        y_norm = 2.0*(foil_coords[:, 1] - ymin) / (ymax - ymin) - 1.0
        
        # 2. Reshape for grid_sample: [Batch, N_points, 1, XY]
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).unsqueeze(2)
        grid = grid.to(self.device) 
        
        # 3. Sample the Cp field (index 2)
        # This acts as your "extrapolation" to the exact surface coordinates
        cp_field = y_pred[:, 2:3, :, :]
        surface_cp = F.grid_sample(cp_field, grid, mode='bilinear', align_corners=True)
        
        return surface_cp.flatten() # Values of Cp on the foil skin


    def calculate_cl_from_cp_sdf(self, surface_cp, foil_coords, sdf_grid, grid_bounds, aoa_deg):
        """
        surface_cp: Tensor [N] - Interpolated Cp values at foil_coords
        foil_coords: Tensor [N, 2] - Physical X, Y coordinates
        sdf_grid: Tensor [1, 1, H, W] - The SDF input channel
        grid_bounds: [xmin, xmax, ymin, ymax]
        aoa_deg: float - Angle of Attack
        """


        # 1. Compute Normals from SDF Grid via Gradients
        # Note: torch.gradient returns (grad_dim0, grad_dim1)
        grad_row, grad_col = torch.gradient(sdf_grid.squeeze())
        grad_x = grad_col.to(self.device)
        grad_y = -grad_row.to(self.device) # FLIP THE SIGN HERE
        
        # 2. Sample these Normals at foil_coords using grid_sample
        # (Reuse the normalization logic from our Cp sampling)
        xmin, xmax, ymin, ymax = grid_bounds
        x_norm = 2.0 * (foil_coords[:, 0] - xmin) / (xmax - xmin) - 1.0
        y_norm = 2.0 * (foil_coords[:, 1] - ymin) / (ymax - ymin) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).unsqueeze(2)

        grid=grid.to(self.device)
        
        # Sample grad_x and grad_y onto the foil points
        nx_sampled = F.grid_sample(grad_x.unsqueeze(0).unsqueeze(0), grid, align_corners=True).flatten().to(self.device)
        ny_sampled = F.grid_sample(grad_y.unsqueeze(0).unsqueeze(0), grid, align_corners=True).flatten()
        
        # Normalize the vectors (make them unit length)
        mag = torch.sqrt(nx_sampled**2 + ny_sampled**2 + 1e-8)
        nx, ny = nx_sampled / mag, ny_sampled / mag

        # 3. Integration Logic
        # Calculate segment lengths (ds) between foil points
        dx_seg = foil_coords[1:, 0] - foil_coords[:-1, 0]
        dy_seg = foil_coords[1:, 1] - foil_coords[:-1, 1]
        ds = torch.sqrt(dx_seg**2 + dy_seg**2)
        
        # Midpoint values for Cp and Normals
        cp_mid = 0.5 * (surface_cp[1:] + surface_cp[:-1])
        nx_mid = 0.5 * (nx[1:] + nx[:-1])
        ny_mid = 0.5 * (ny[1:] + ny[:-1])
        
        # 4. Sum Pressure Forces (Force = -Cp * Normal * Length)
        fx = torch.sum(-cp_mid * nx_mid * ds)
        fy = torch.sum(-cp_mid * ny_mid * ds)
        
        # 5. Rotate to Wind Frame (Lift/Drag)
        alpha = torch.tensor(aoa_deg * torch.pi / 180.0).to(self.device)
        clp = fy * torch.cos(alpha) - fx * torch.sin(alpha)
        
        return clp.item()
    
    def sort_airfoil_points(coords, data_values):
        """
        Sorts airfoil coordinates and associated data (like Cp) 
        by their angular position around the centroid.
        """
        # 1. Calculate the centroid of the points
        centroid_x = torch.mean(coords[:, 0])
        centroid_y = torch.mean(coords[:, 1])

        # 2. Calculate the angle for each point
        angles = torch.atan2(coords[:, 1] - centroid_y, coords[:, 0] - centroid_x)

        # 3. Get sort indices
        sort_idx = torch.argsort(angles)

        return coords[sort_idx], data_values[sort_idx]

    def plot_surface_diagnostic(self, y_phys, batch, sample_idx=0, save_dir="surface_plots"):
        """Plots the Cp distribution vs x/c for a single airfoil"""
        batch_props = batch['props']
        props = batch_props[sample_idx]
        foil_coords = props['airfoil_points'].to(self.device)

        
        cp_surf = self.get_surface_cp_on_foil(y_phys[sample_idx:sample_idx+1], foil_coords, props['grid_bounds'])


        sorted_coords, sorted_cp = self.sort_airfoil_points(foil_coords.cpu(), cp_surf.cpu())

        clp = self.calculate_cl_from_cp_sdf(
                surface_cp=cp_surf,
                foil_coords=foil_coords,
                sdf_grid=batch['x'][sample_idx:sample_idx+1, 2:3].to(self.device), 
                grid_bounds=props['grid_bounds'],
                aoa_deg=props['aoa_deg']
            )    
          
        # Move to CPU for plotting
        x_coords = sorted_coords[:, 0].numpy()
        cp_vals = sorted_cp.numpy()
        
        plt.figure(figsize=(10, 6))
        # Plotting -Cp is the aerodynamic standard (suction is 'up')
        plt.plot(x_coords, -cp_vals, 'r-', label='FNO Prediction', linewidth=2)
        
        # If truth values exist in props, plot them as dots
        if 'cp_truth' in props:
             plt.scatter(props['airfoil_points'][:, 0], -props['cp_truth'], 
                         c='k', s=10, alpha=0.5, label='CFD Truth')
        clp_truth = props.get('clp', None)
        if clp_truth is not None:
            plt.title(f"Surface Pressure Distribution - AoA: {props['aoa_deg']}°   Clp: {clp:.4f} (Truth: {clp_truth:.4f})")
        else:
            plt.title(f"Surface Pressure Distribution - AoA: {props['aoa_deg']}°   Clp: {clp:.4f}    ")

        plt.gca().invert_yaxis() 
        plt.xlabel('x/c (Chord Position)')
        plt.ylabel('-Cp (Pressure Coefficient)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        outpufile_dir = Path(f"{save_dir}/surface")
        outpufile_dir.mkdir(parents=True, exist_ok=True)  
        plt.savefig(f"{outpufile_dir}/surface_cp_sample{sample_idx}.png")
        plt.close()
        return clp_truth,clp