import torch
import sys
import os
import neuralop
from neuralop.models import FNO
# Ensure you import your custom DataProcessor if it's not a standard neuralop one
# from your_module import AirfransDataProcessor 
import matplotlib.pyplot as plt
from pathlib import Path
from zencfg import make_config_from_cli
import numpy as np
from tqdm import tqdm

from neuralop.models.base_model import get_model
sys.path.insert(0, "../")
from zencfg import make_config_from_cli
from tims.airfransX5Y4.airfrans_datasetX5Y4_v1 import load_airfrans_dataset, get_dataset_stats
from tims.airfransX5Y4.config_AirfransX5Y4_v1 import Default
import torch.nn.functional as F
import csv

torch.serialization.add_safe_globals([torch._C._nn.gelu, neuralop.layers.spectral_convolution.SpectralConv])

class ModelPredictor:

    def __init__(self, model_path,state_dict_path, processor_path, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        # Model initialization

        model = get_model(config)
        # Load model state dict
        state_dict = torch.load(state_dict_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()


        # Move model to device
        model = model.to(self.device)
        self.model = model
        print(f"Model moved to device: {self.device}")
        # Load the Airfrans dataset
        self.data_dir = Path(config.data.data_dir).expanduser()

        train_loader, test_loaders, data_processor_dataset = load_airfrans_dataset(
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
        self.train_loader = train_loader
        self.test_loaders = test_loaders
        self.data_processor = data_processor_dataset

        if processor_path is not None:
            processor_state_dict = torch.load(processor_path, map_location=self.device)
            self.data_processor.load_state_dict(processor_state_dict)
            self.data_processor = self.data_processor.to(self.device)
            self.data_processor.eval()   

            print("Data processor path provided. Ensure input data is preprocessed correctly.")
        # check data_processor
        print(" =" * 80)
        self.verify_input_encoder(self.data_processor.in_normalizer)

        self.verify_output_encoder(self.data_processor.out_normalizer)
        print(" =" * 80)            

    def verify_input_encoder(self, encoder):
        print(f"\n{'='*20} INPUT ENCODER AUDIT {'='*20}")
        if encoder is None:
            print("No input encoder detected. Skipping audit.")
            return
        # 1. Check Channel Dimensions
        mean = encoder.mean.flatten()
        std = encoder.std.flatten()
        print(f"Stats Shape: {list(encoder.mean.shape)} | Channels detected: {len(mean)}")

        # 2. Check Physical Mapping
        # We expect 5 channels of stats representing [u_inf, v_inf, mask, sdf, log_Re]
        # Mask should be unaltered min=0, max=1
        names = ["u_velocity (inf)", "v_velocity (inf)", "mask", "SDF (geometry)", "log_Re"]
        
        print(f"\n{'Channel':<20} | {'Mean':>10} | {'Std':>10}")
        print("-" * 45)
        for i, name in enumerate(names):
            m, s = mean[i].item(), std[i].item()
            print(f"{name:<20} | {m:>10.4f} | {s:>10.4f}")

        # 3. Verify Selective Logic
        channels = getattr(encoder, 'channels_to_normalize', [])
        print(f"\nActive Channels for Normalization: {channels}")
        
        if 2 in channels:
            print("!! WARNING: Channel 2 (Mask) is set to be normalized! This will corrupt geometry.")
        else:
            print("✓ SUCCESS: Channel 2 (Mask) will be passed through untouched.")
        print(f"{'='*63}\n")

    def verify_output_encoder(self, encoder):
        print(f"\n{'='*20} OUTPUT ENCODER AUDIT {'='*20}")
        if encoder is None:
            print("No output encoder detected. Skipping audit.")
            return
        # 1. Check Channel Dimensions
        mean = encoder.mean.flatten()
        std = encoder.std.flatten()
        print(f"Stats Shape: {list(encoder.mean.shape)} | Channels detected: {len(mean)}")

        # 2. Check Physical Mapping
        # We expect 4 channels of stats representing [u_deficit, v_deficit, Cp, log_nut_ratio]
        # which will be applied to indices [0, 1, 2, 3] of the 4D output.
        names = ["u_deficit", "v_deficit", "Cp", "log_nut_ratio"]
        
        print(f"\n{'Channel':<20} | {'Mean':>10} | {'Std':>10}")
        print("-" * 45)
        for i, name in enumerate(names):
            m, s = mean[i].item(), std[i].item()
            print(f"{name:<20} | {m:>10.4f} | {s:>10.4f}")

        print(f"{'='*63}\n")


    def get_sample(self, test=False, test_loader_idx=0, index=0):


        # 1. Fetch the specific sample
        if test:
            loader = self.train_loader
            res = self.config.data.train_resolution
            split = self.config.data.train_split
            print(f"Using training loader for plotting with resolution {res} and split '{split}'.")
        else:
            loader = list(self.test_loaders.values())[test_loader_idx]
            res = self.config.data.test_resolutions[test_loader_idx]
            split = self.config.data.test_splits[test_loader_idx]
            print(f"Using test loader index {test_loader_idx} for plotting with resolution {res} and split '{split}'.")


        self.current_index = index
        self.current_res = res
        self.current_split = split

        x_raw, y_raw, props = None, None, None
        for i, batch in enumerate(loader):
            if i == index // loader.batch_size:
                idx_in_batch = index % loader.batch_size
                x_raw = batch['x'][idx_in_batch].unsqueeze(0).to(self.device)
                y_raw = batch['y'][idx_in_batch].unsqueeze(0).to(self.device)
                raw_props = batch['props']
                
                if isinstance(raw_props, list):
                    # If it's a list of dicts (common with custom collate)
                    props = raw_props[idx_in_batch]
                elif isinstance(raw_props, dict):
                    # If it's a dict of tensors (standard PyTorch collation)
                    # We rebuild a single dictionary for this specific sample
                    props = {k: v[idx_in_batch] for k, v in raw_props.items()}
                
                break
        return x_raw, y_raw, props,res, split
    

    def get_props_info(self, test=False, test_loader_idx=0, index=0):
        # Fetch sample tensors and property dictionary
        x_raw, y_raw, props , res, split= self.get_sample(test=test, test_loader_idx=test_loader_idx, index=index)

        if props is None:
            print("No properties found for this sample.")
            return None

        print(f"\n=== Metadata Audit: Sample {index} ===")
        for key, value in props.items():
            if torch.is_tensor(value):
                # Print shape and mean for tensors (like airfoil_points)
                print(f"{key:15} | Shape: {list(value.shape)} | Mean: {value.mean().item():.4f}")
            else:
                # Print raw value for scalars (like aoa_deg)
                print(f"{key:15} | Value: {value}")
        
        return props

    def load_and_predict(self, test=False, test_loader_idx=0,  index=0):
        # Fetch batch
        x_raw, y_raw, props , res, split= self.get_sample(test=test, test_loader_idx=test_loader_idx, index=index)
            
        if x_raw.dim() == 3:
            x_raw = x_raw.unsqueeze(0)  # Add batch dimension
        if y_raw.dim() == 3:
            y_raw = y_raw.unsqueeze(0)  # Add batch dimension
        # 4. Predict
        with torch.no_grad():
            # Preprocess: handles normalization and device transfer
            # input_data should be a dict: {'x': tensor, 'y': tensor_or_none}
            sample = self.data_processor.preprocess({'x': x_raw, 'y': y_raw})
            # transfer all tensors to device
            sample = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in sample.items()}

            model_input = {k: v for k, v in sample.items() if k != 'y'}
            
            # Forward pass
            output = self.model(**model_input)
            
            # Postprocess: Inverts normalization (e.g., back to m/s and pressure)
            # and returns physical units
            decoded_output, _= self.data_processor.postprocess(output, sample)

        return decoded_output

    def plot_validation(self, test=False, test_loader_idx=0, index=0, decoded=True):
        # Fetch batch
        x_raw, y_raw, props , res, split= self.get_sample(test=test, test_loader_idx=test_loader_idx, index=index)  
        with torch.no_grad():
            # 2. Preprocess & Predict
            sample = self.data_processor.preprocess({'x': x_raw, 'y': y_raw})
            sample = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in sample.items()}
            
            model_input = {k: v for k, v in sample.items() if k != 'y'}
            output = self.model(**model_input)
            
            # 3. Handle Toggle: Decoded (Physical) vs Encoded (Normalized)
            if decoded:
                # Postprocess returns physical units
                res_pred, _ = self.data_processor.postprocess(output, sample)
                y_p = res_pred['y'].squeeze(0).cpu()
                y_t = y_raw.squeeze(0).cpu()
                mode_title = "Decoded "
            else:
                # Use raw model output and normalized sample['y']
                y_p = output.squeeze(0).cpu()
                y_t = sample['y'].squeeze(0).cpu()
                mode_title = "Encoded "

        # 4. Calculate Residuals (Absolute Difference)
        y_residual = torch.abs(y_p - y_t)

        # 5. Plotting (3 Rows x 4 Channels)
        fig, axes = plt.subplots(3, 4, figsize=(22, 14))
        out_titles = ["U-Velocity", "V-Velocity", "Cp (Pressure)", "log(nut_ratio)"]
        cmaps = ['viridis', 'viridis', 'plasma', 'magma']

        for i in range(4):
            # Shared scale for Pred and Truth
            vmin = min(y_t[i].min(), y_p[i].min())
            vmax = max(y_t[i].max(), y_p[i].max())

            # Row 0: Prediction
            im0 = axes[0, i].imshow(y_p[i], origin='lower', cmap=cmaps[i], vmin=vmin, vmax=vmax)
            axes[0, i].set_title(f"Pred: {out_titles[i]}")
            fig.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)

            # Row 1: Truth
            im1 = axes[1, i].imshow(y_t[i], origin='lower', cmap=cmaps[i], vmin=vmin, vmax=vmax)
            axes[1, i].set_title(f"Truth: {out_titles[i]}")
            fig.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)

            # Row 2: Residuals (|True - Pred|)
            # Use a 'hot' or 'inferno' map for errors to highlight spikes
            im2 = axes[2, i].imshow(y_residual[i], origin='lower', cmap='inferno')
            axes[2, i].set_title(f"Abs. Residual: {out_titles[i]}")
            fig.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)

        plt.suptitle(f"AirFrans Validation | {mode_title} | Resolution {res} | {split} | Index: {index}", fontsize=20, y=0.98)
        plt.tight_layout()
        plt.show()

    def sort_airfoil_chain(self, coords):
        """
        Guarantees a smooth loop by always picking the next closest point.
        coords: [N, 2] tensor
        """
        unvisited = list(range(len(coords)))
        # Start at the Trailing Edge (max X)
        curr_idx = torch.argmax(coords[:, 0]).item()
        sorted_idx = [curr_idx]
        unvisited.remove(curr_idx)

        while unvisited:
            # Find the single closest point to the current one
            dists = torch.norm(coords[unvisited] - coords[curr_idx], dim=1)
            nearest_neighbor_idx = torch.argmin(dists).item()
            
            curr_idx = unvisited[nearest_neighbor_idx]
            sorted_idx.append(curr_idx)
            unvisited.pop(nearest_neighbor_idx)

        return coords[sorted_idx]
        


    def pad_airfoil_interior_priority(self, y_phys_tensor, x_raw, iterations=5):
        """
        Surgically extrudes surface pressure into the interior.
        SDF Convention: Negative = Flow, Positive = Inside Airfoil
        """
        y_phys_buffered = y_phys_tensor.clone()

        sdf = x_raw[:, 3:4, :, :]  # Extract SDF channel
        
        # 1. Define the 'Flow' mask: In your case, Flow is where SDF < 0
        flow_mask = (sdf < 0).float()
        
        # 2. Iterative 'Push'
        # Create kernel for flow_mask (1 channel)
        mask_kernel = torch.ones((1, 1, 3, 3)).to(self.device)
        # Create kernel for y_phys_buffered (4 channels, processed independently)
        phys_kernel = torch.ones((4, 1, 3, 3)).to(self.device)
        
        print(f" Pushing surface values into the interior for {iterations} iterations...")
        print(f"Initial flow_mask sum (should be total flow pixels): {flow_mask.sum().item()}")
        
        for i in range(iterations):
            # Calculate neighbor support using single-channel mask
            sum_mask = F.conv2d(flow_mask, mask_kernel, padding=1) + 1e-8
            
            # Calculate the smeared/average for all 4 channels using grouped convolution
            y_smeared = F.conv2d(y_phys_buffered * flow_mask, phys_kernel, padding=1, groups=4) / sum_mask
            
            # Update Zone: Where it's currently NOT flow (SDF > 0) 
            # but touches the flow (sum_mask > 0)
            update_zone = (flow_mask < 0.5) & (sum_mask > 0.1)
            
            # Apply the update
            y_phys_buffered = torch.where(update_zone, y_smeared, y_phys_buffered)
            
            # Expand the flow_mask so the next iteration can push further inside
            flow_mask = torch.where(update_zone, torch.ones_like(flow_mask), flow_mask)
            print(f"Iteration {i+1}/{iterations} | Updated pixels: {update_zone.sum().item()} | Total flow_mask sum: {flow_mask.sum().item()}")

        return y_phys_buffered


    
    def sample_Y_on_airfoil(self, y_phys, channel, airfoil_coords, grid_bounds):
        """
        Interpolates Y values from the FNO grid at specific airfoil coordinates.
        
        y_phys: [1, 4, H, W] - Physical FNO output (Channel 0: u, 1: v, 2: Cp, 3: log(nut_ratio))
        airfoil_coords: [N, 2] - Sorted (x, y) coordinates from props
        grid_bounds: [xmin, xmax, ymin, ymax]
        """
        
        xmin, xmax, ymin, ymax = grid_bounds.to(self.device)

        xmin.to(self.device)
        xmax.to(self.device)
        ymin.to(self.device)
        ymax.to(self.device)

        airfoil_coords = airfoil_coords.to(self.device)
        # 1. Normalize coordinates to [-1, 1] for grid_sample
        # This maps your physical coordinates to the FNO's grid index space
        x_norm = 2.0 * (airfoil_coords[:, 0] - xmin) / (xmax - xmin) - 1.0
        y_norm = 2.0 * (airfoil_coords[:, 1] - ymin) / (ymax - ymin) - 1.0
        
        # Reshape for grid_sample: [Batch, N_points, 1, 2]
        grid_input = torch.stack([x_norm, y_norm], dim=-1)  # Shape: [1, N, 1, 2]
        grid_query = grid_input.unsqueeze(0).unsqueeze(2)
        
        # 2. Extract the specified channel
        y_grid = y_phys[:, channel:channel+1, :, :] # Shape: [1, 1, H, W]
        
        # 3. Interpolate using bilinear sampling
        # align_corners=True ensures the edges of your grid match the bounds
        y_sampled = torch.nn.functional.grid_sample(
            y_grid, 
            grid_query.to(y_grid.device), 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        plot=False
        if plot:
            print(f"Debug: Plotting Channel {channel} interpolation at airfoil points...")
            plt.figure(figsize=(8, 4))
            plt.plot(x_norm.cpu(), y_norm.cpu(), 'k.-', markersize=2, label='Airfoil Perimeter')
            plt.plot(grid_input[:, 0].cpu(), grid_input[:, 1].cpu(), 'bx', label='Normalized Airfoil Points')
            plt.imshow(y_grid[0, 0].cpu(), origin='lower', extent=(-1, 1, -1, 1), cmap='viridis', alpha=0.5)

            #plt.scatter(airfoil_coords[:, 0].cpu(), airfoil_coords[:, 1].cpu(), c=cp_sampled.flatten().cpu(), cmap='plasma', label='Interpolated Cp')
            plt.colorbar(label='Y Value')
            plt.axis('equal')
            plt.title(f" Y Channel {channel} Grid Airfoil Points")
            plt.legend()
            plt.show()
        
        return y_sampled.flatten() # Returns [1007] tensor of Y values
       
    
    def get_normals_tangents_from_sdf(self, x_raw, airfoil_coords, grid_bounds):
        """
        Extracts outward-facing normals using the SDF gradient.
        SDF convention: Negative (flow) to Positive (inside).
        """
        # 1. Extract SDF Channel (Index 3)
        sdf = x_raw[:, 3:4, :, :] # [1, 1, H, W]
        
        # 2. Calculate Spatial Gradients
        # Use torch.gradient or Sobel filters. 
        # dy is the 0th dim (rows), dx is the 1st dim (cols).
        grad_y, grad_x = torch.gradient(sdf.squeeze(0).squeeze(0))
        
        # 3. Stack and Normalize for Sampling: [1, 2, H, W]
        grad_field = torch.stack([grad_x, grad_y], dim=0).unsqueeze(0)

        # 4. Map Coordinates to [-1, 1] for grid_sample
        xmin, xmax, ymin, ymax = grid_bounds
        # Ensure airfoil_coords is on the same device as grid_bounds
        airfoil_coords = airfoil_coords.to(self.device)
        x_norm = 2.0 * (airfoil_coords[:, 0] - xmin) / (xmax - xmin) - 1.0
        y_norm = 2.0 * (airfoil_coords[:, 1] - ymin) / (ymax - ymin) - 1.0
        grid_query = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).unsqueeze(2)

        # 5. Sample Gradient at Airfoil Points
        sampled_grad = torch.nn.functional.grid_sample(
            grad_field, grid_query.to(self.device), align_corners=True
        ).squeeze().T # Result: [N, 2]

        # 6. Final Outward Normal: 
        # Gradient points IN (toward positive SDF). We want OUT.
        outward_normals = -sampled_grad 
        outward_normals = outward_normals / torch.norm(outward_normals, dim=1, keepdim=True)

        #  Get the tangent vectors for viscous flow
        # must point in the direction of flow (from LE to TE on upper, LE to TE on lower)
        tangents_downstream = torch.stack([outward_normals[:, 1], -outward_normals[:, 0]], dim=1)
        flip_mask = (tangents_downstream[:, 0] < 0)
        tangents_downstream[flip_mask] = -tangents_downstream[flip_mask]
        
        return outward_normals, tangents_downstream
    
    def calculate_clp_from_surface(self, cp_values, sorted_coords,normals_out, props):
        """
        Integrates pressure along the airfoil perimeter to find the Lift Coefficient.
        
        cp_values: [N] tensor of stabilized Cp values
        sorted_coords: [N, 2] tensor of ordered (x, y) coordinates
        props: dictionary containing 'aoa_deg'
        """
        # 1. Calculate segment lengths and directions (dx, dy)
        # Roll to get P_{i+1} - P_i (closes the loop automatically)
        sorted_coords = sorted_coords.to(self.device)  # Ensure on CPU for easier manipulation
        p_next = torch.roll(sorted_coords, -1, dims=0)
        ds = torch.norm(p_next - sorted_coords, dim=1) # Length of each segment
        
        # 2. Convert AoA to radians
        aoa_rad = torch.tensor(props['aoa_deg'] * (torch.pi / 180.0)).to(self.device)
                
        # Average Cp on each segment for trapezoidal accuracy
        cp_avg = 0.5 * (cp_values + torch.roll(cp_values, -1, dims=0))
        n_avg =0.5 * (normals_out + torch.roll(normals_out, -1, dims=0))
        
        # Force = -Cp * Normal * ds
        # This ensures suction (negative Cp) pulls OUTWARD.
        df_x = -cp_avg * n_avg[:, 0] * ds
        df_y = -cp_avg * n_avg[:, 1] * ds
        
        #print(f" Debug: cp_avg , fx and fy shapes: {cp_avg.shape}, {df_x.shape}, {df_y.shape}")
        #print(f" Debug: AoA (deg): {props['aoa_deg']} | AoA (rad): {aoa_rad.item():.4f}")
        
        # 3. Resolve into Lift and Drag components  aligned with freestream
        lift_segments = df_y * torch.cos(aoa_rad) - df_x * torch.sin(aoa_rad)
        drag_segments = df_x * torch.cos(aoa_rad) + df_y * torch.sin(aoa_rad)

        # 4. Integrate over the surface
        clp = torch.sum(lift_segments)
        cdp = torch.sum(drag_segments)
        print(f" Aoa,  Truth CLp ,   Pred Clp,     Truth CDp ,   Pred CDp ")
        print(f" {props['aoa_deg']:.4f} , {props['clp'].item():.4f} , {clp.item():.4f} , {props['cdp'].item():.4f} , {cdp.item():.4f} ")

        results_dict = {
            'clp': clp.item(),
            'cdp': cdp.item(),
            'truth_clp': props['clp'].item(),
            'truth_cdp': props['cdp'].item()
        }
        return results_dict

    def estimate_tau_w_wall_function(self, u_parallel, y_dist, props):
        """
        Estimates Tau_w using the Log-Law of the Wall.
        u_parallel: Velocity parallel to surface at height y_dist
        y_dist: Distance from wall assumes 1 -pixel size
        """
        rho = props['rho'].item() 
        nu = props['nu_mol'].item() 
        grid_bounds = props['grid_bounds']
        x_range = grid_bounds[1] - grid_bounds[0]
        y_range = grid_bounds[3] - grid_bounds[2]
        
        kappa = 0.41
        B = 5.2

        
        # Initial guess for u_tau using a simplified friction factor
        u_mag = torch.abs(u_parallel) + 1e-8  # Avoid zero velocity
        u_tau = 0.05 * u_mag 
        
        # 3-5 iterations are usually enough for convergence
        for _ in range(5):
            y_plus = (y_dist * u_tau) / nu
            yplus_clipped = torch.clamp(y_plus, min=1e-8, max=1e5)  # Avoid log(0) and excessively large values
            # Log-law: u_tau = u_parallel / ( (1/kappa)*ln(y_plus) + B )
            u_tau = u_mag / ((1/kappa) * torch.log(yplus_clipped) + B)

            mean_u_tau = u_tau.mean().item()
            mean_y_plus = y_plus.mean().item()
            print(f" Iteration {_+1}: Mean u_tau: {mean_u_tau:.4f}, Mean y_plus: {mean_y_plus:.4f} , Mean u_parallel: {u_mag.mean().item():.4f}")
            
        tau_w = rho * (u_tau**2)
        return tau_w, y_plus
    
    def estimate_viscous_forces(self, u_values, v_values, res, props,sorted_coords,tangents_downstream, grid_bounds, save_dir=None,export=False):
        """
        Estimates wall shear stress (tau_w) using velocity gradients at the airfoil surface.
        SDF convention: Negative (flow) to Positive (inside).
        """
        v_mag_inf =props['v_mag_inf'].to(self.device)
        aoa_rad = torch.tensor(props['aoa_deg'] * (torch.pi / 180.0)).to(self.device)

        u_inf = v_mag_inf * torch.cos(aoa_rad)
        v_inf = v_mag_inf * torch.sin(aoa_rad)

        # convert back to absolute velocities
        u_phys= -(u_values * v_mag_inf - u_inf)
        v_phys= -(v_values * v_mag_inf - v_inf)

        #tangential directions
        tx, ty = tangents_downstream[:, 0], tangents_downstream[:, 1]

        # estimated tangential velocity near the wall using the velocity field
        u_parallel = (u_phys * tx) + (v_phys * ty)


        print(f" Export {export} | Save Dir: {save_dir}")
        if export and save_dir is not None:
            print(f"Debug: u_parallel stats - min: {u_parallel.min().item():.4f}, max: {u_parallel.max().item():.4f}, mean: {u_parallel.mean().item():.4f}")
            print(f"Debug: Grid resolution (res): {res} | Grid bounds: {grid_bounds.cpu().numpy()}")
            plt.figure(figsize=(12, 6))
            
            # Left subplot: u_parallel colored scatter
            plt.subplot(1, 2, 1)
            plt.plot(sorted_coords[:, 0].cpu(), sorted_coords[:, 1].cpu(), 'k.-', markersize=2, label='Airfoil')
            plt.scatter(sorted_coords[:, 0].cpu(), sorted_coords[:, 1].cpu(), c=u_parallel.cpu(), cmap='viridis', label='u_parallel')
            plt.colorbar(label='u_parallel (m/s)')
            plt.title("Estimated Tangential Velocity (u_parallel) at Airfoil Surface")
            plt.axis('equal')
            
            # Right subplot: velocity vectors
            plt.subplot(1, 2, 2)
            plt.plot(sorted_coords[:, 0].cpu(), sorted_coords[:, 1].cpu(), 'k.-', markersize=2, label='Airfoil')
            
            # Normalize velocity vectors to 0.01 * v_mag_inf for visualization
            scale_factor = 0.1
            u_scaled = (u_phys / v_mag_inf * scale_factor).cpu()
            v_scaled = (v_phys / v_mag_inf * scale_factor).cpu()
            
            plt.quiver(sorted_coords[:, 0].cpu(), sorted_coords[:, 1].cpu(), 
                      u_scaled, v_scaled, 
                      angles='xy', scale_units='xy', scale=1, 
                      color='red', alpha=0.7, width=0.002)
            

            plt.title(f"Velocity Vectors at Airfoil Surface\n(Scaled to {scale_factor:.3f} × v_inf)")
            plt.axis('equal')
            export_path = save_dir / "vectors" 
            export_file = Path(export_path ) / f"velocity_vectors_index_{self.current_index}_res_{self.current_res}_{self.current_split}.png"
            os.makedirs(os.path.dirname(export_file), exist_ok=True)
            plt.savefig(export_file, dpi=300)

        # Assume y_dist is the grid spacing (you can refine this by calculating actual distance to wall)
        grid_bounds = props['grid_bounds'].to(self.device)
        x_range = grid_bounds[1] - grid_bounds[0]
        y_range = grid_bounds[3] - grid_bounds[2]

        grid_size_x = res[0]
        grid_size_y = res[1]
        dx = x_range / grid_size_x
        dy = y_range / grid_size_y
        y_dist = min(dx, dy) 

        # Conservative estimate of distance to wall
        print(f"Estimated y_dist for wall function: {y_dist:.6f} (based on grid spacing)")
        tau_w, y_plus = self.estimate_tau_w_wall_function(u_parallel, y_dist, props)
        
        # integrate tau_w along the surface to get viscous drag contribution
        p_next = torch.roll(sorted_coords, -1, dims=0)
        ds = torch.norm(p_next - sorted_coords, dim=1).to(self.device) # Length of each segment  
        # 3. Local Viscous Forces
        # Average tau_w on the segment for trapezoidal accuracy
        tau_avg = 0.5 * (tau_w + torch.roll(tau_w, -1, dims=0)).to(self.device)
        df_vx = tau_avg * tx * ds 
        df_vy = tau_avg * ty * ds
        cos_a, sin_a = torch.cos(aoa_rad), torch.sin(aoa_rad)
        df_lift = df_vy * cos_a - df_vx * sin_a
        df_drag = df_vx * cos_a + df_vy * sin_a
        clv = torch.sum(df_lift) / (0.5 * props['rho'].item() * (props['v_mag_inf'].item()**2) )
        cdv = torch.sum(df_drag) / (0.5 * props['rho'].item() * (props['v_mag_inf'].item()**2) )
        return clv.item(), cdv.item()       


    def plot_airfoil_cp(self, test=False, test_loader_idx=1, index=13, save_dir=None,export=False):
        # 1. Fetch sample and properties
        x_raw, y_raw, props , res, split= self.get_sample(test=test, test_loader_idx=test_loader_idx, index=index)


        
        # 2. Extract and slice coordinates: [1007, 3] -> [1007, 2]
        airfoil_coords = props['airfoil_points'][:, :2].cpu()
               

        sorted_coords = self.sort_airfoil_chain(airfoil_coords)

        # get Cp values at these coordinates
        y_phys_dict = self.load_and_predict(test=test, test_loader_idx=test_loader_idx, index=index)
        y_phys = y_phys_dict['y']  # [4, H, W]
        
        grid_bounds = props['grid_bounds'].to(self.device)
        #mask = 1 - x_raw[0, 2:3, :, :]  # Assuming channel 2 is the mask |1.0 for flow, 0.0 for airfoil 

        # 4. Pad the interior of the airfoil to get better Cp sampling
        y_buffered = self.pad_airfoil_interior_priority(y_phys, x_raw , iterations=2)  
        
        # get u, v, and Cp values at the airfoil coordinates
        u_values = self.sample_Y_on_airfoil(y_buffered, 0, sorted_coords, grid_bounds)
        v_values = self.sample_Y_on_airfoil(y_buffered, 1, sorted_coords, grid_bounds)
        cp_values = self.sample_Y_on_airfoil(y_buffered, 2, sorted_coords, grid_bounds)


        # get outward normals
        normals_out,tangents_downstream = self.get_normals_tangents_from_sdf(x_raw, sorted_coords, grid_bounds)

        # calcuate CLp and CDp from surface integration
        results_dict = self.calculate_clp_from_surface(cp_values, sorted_coords, normals_out, props)

        # get grid resolution for wall function calculations
        res_xy =y_phys.shape[2], y_phys.shape[3] 

        # estimate viscous forces and get CLv and CDv
        clv, cdv = self.estimate_viscous_forces(u_values, v_values, res_xy, props, sorted_coords, tangents_downstream, grid_bounds,save_dir=save_dir,export=export   )
        results_dict['clv'] = clv
        results_dict['cdv'] = cdv   
        
        # 5. Plotting
        plt.figure(figsize=(8, 4))
        plt.plot(sorted_coords[:, 0], -sorted_coords[:, 1], 'k.-', markersize=2, label='Airfoil')
        
        # Plot Cp 
        plt.plot(sorted_coords[:, 0].cpu(), cp_values.cpu(), 'r-', label='Cp Prediction')
        # set plot limits
        plt.xlim(-1, 2)

        plt.ylim(-2, 1)
        # reverse y-axis for Cp convention (lower Cp is higher pressure, so it should be "up" on the plot)
        plt.gca().invert_yaxis()


        #plt.axis('equal')

        if test:
            loader_info = f" Res {res} {split}"
        else:
            loader_info = f" Res {res} {split}"
            
        plt.title(f"Airfoil {loader_info} - Index {index} Aoa {props['aoa_deg']:.4f} deg \n CLp: {results_dict['clp']:.4f} (Truth: {results_dict['truth_clp']:.4f}), CDp: {results_dict['cdp']:.4f} (Truth: {results_dict['truth_cdp']:.4f})")
        plt.legend()
        if export and save_dir is not None:
            export_dir = Path(save_dir) / "surface_cp"
            os.makedirs(export_dir, exist_ok=True)  
            export_png_path = export_dir / f"airfoil_{loader_info.replace(' ', '_')}_index_{index}.png"
            plt.savefig(export_png_path, dpi=300)
            print(f"Plot exported to {export_png_path}")

            results_dict = {
                'train_index': index,
                'aoa_deg': props['aoa_deg'].item(),
                'v_mag_inf': props['v_mag_inf'].item(),
                'reynolds': props['reynolds'].item(),
                'truth_cl': props['clp'].item(),
                'truth_cd': props['cd'].item(),
                'truth_clp': props['clp'].item(),
                'truth_cdp': props['cdp'].item(),
                'truth_clv': props['clv'].item(),
                'truth_cdv': props['cdv'].item(),
                'clp': results_dict['clp'],
                'cdp': results_dict['cdp'],
                'clv': results_dict['clv'],
                'cdv': results_dict['cdv']
            }
            fmt = "{:>12.4f}" 
            # Create a formatted version of row_data
            formatted_row = {}
            for key, value in results_dict.items():
                if key == 'reynolds':
                    formatted_row[key] = "{:>12d}".format(int(value))

                elif isinstance(value, (float, int)):
                    formatted_row[key] = fmt.format(value)
                else:
                    formatted_row[key] = value
            # 5. CSV Logging
            log_file = f"{save_dir}/airfoil_{res}_{split}_results.csv" 
            file_exists = os.path.isfile(log_file)
            with open(log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['train_index',
                                                       'aoa_deg', 
                                                       'v_mag_inf', 
                                                       'reynolds', 
                                                       'truth_cl',
                                                       'truth_cd',
                                                       'truth_clp',
                                                       'truth_cdp',
                                                       'truth_clv',
                                                       'truth_cdv', 
                                                       'clp',
                                                       'cdp',
                                                       'clv',
                                                       'cdv'])
                if not file_exists:
                    writer.writeheader()
                writer.writerow(formatted_row)        


if __name__ == "__main__":

    # Usage
    ROOT_DIR = Path("/home/timm/Projects/PIML/neuraloperator/tims/airfransX5Y4/test_model")
    MODEL_PATH = ROOT_DIR / 'model_metadata.pkl'
    STATE_DICT_PATH = ROOT_DIR / 'model_state_dict.pt'
    DATA_PROCESSOR_PATH = ROOT_DIR / 'data_processor.pt'

    EXPORT_DIR =Path("/home/timm/Projects/PIML/neuraloperator/tims/airfransX5Y4/test_model")

    # Example usage
    config = make_config_from_cli(Default)
    config = config.to_dict()

    predictor = ModelPredictor(MODEL_PATH, STATE_DICT_PATH, DATA_PROCESSOR_PATH, config)

   
    y_out = predictor.load_and_predict(test=False, test_loader_idx=0, index=0)

    predictor.get_props_info(test=False, test_loader_idx=0, index=0)
    

    print("Prediction completed.")

    # Sanity check in your script
    for i, name in enumerate(["u", "v", "cp", "log_nutratio"]):
        mean = predictor.data_processor.out_normalizer.mean[0, i, 0, 0].item()
        std = predictor.data_processor.out_normalizer.std[0, i, 0, 0].item()
        print(f"Channel {i} ({name}): Physical = (Encoded * {std:.4f}) + {mean:.4f}")
    
    print ("\nNow plotting Cp distribution at airfoil points...")

    test_loader_idx = 1
    test_loader_resolutions = config.data.test_resolutions
    test_res = test_loader_resolutions[test_loader_idx]
    test_splits = config.data.test_splits
    test_split = test_splits[test_loader_idx]
    loader_info = f"Test Loader - Res {test_res} {test_split}  "
    
    num_samples = len(predictor.test_loaders[list(predictor.test_loaders.keys())[test_loader_idx]].dataset)

    export_dir = ROOT_DIR / f"plots_{test_res}_{test_split}"
    os.makedirs(export_dir, exist_ok=True)
    print(f"{loader_info} contains {num_samples} samples.")
    use_test_loader = False
    for idx in tqdm(range(num_samples), desc=f"Plotting Cp for {loader_info}"):
        #print(f"\nPlotting sample index {idx} from {loader_info}...")    
        predictor.plot_airfoil_cp(use_test_loader, test_loader_idx, index=idx, save_dir=export_dir, export=True)