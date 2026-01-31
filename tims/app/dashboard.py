import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from neuralop.models import FNO # Or your specific GINO/FNO import
import zencfg.bunch
import neuralop
import pickle
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
import sys
import os
from pathlib import Path
from matplotlib.path import Path as mplPath


# Add the tims directory to the path so we can import from airfrans_all_out
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../airfrans_all_out"))

from airfrans_all_out.airfrans_dataset_all import AirfransDatasetAll,SelectiveDataProcessor,SelectiveUnitGaussianNormalizer
from airfrans_all_out.airfrans_dataset_all import load_airfrans_dataset

import torch




@st.cache_resource
def get_initialized_processor():
    """
    Reconstructs the training environment to get the fitted DataProcessor.
    """
    # Use the same parameters as your training script to ensure 
    # normalization stats match
    # Create dataset instance for plotting
    train_loader, test_loaders, data_processor = load_airfrans_dataset(
        data_dir="/home/timm/Projects/PIML/neuraloperator/tims/airfrans/consolidated_Cp_data",
        dataset_name="airfoil_cp",
        train_split="full_train",
        test_splits=['full_test','full_test'],
        batch_size=16,
        test_batch_sizes=[64],
        test_resolutions=[128, 256],
        encode_input=True,    
        encode_output=True, 
        encoding="channel-wise",
        channel_dim=1,
    )
    
    return data_processor.to('cpu'), train_loader


if "data_processor" not in st.session_state:
    with st.spinner("Initializing Physics Engine (Syncing Normalizers)..."):
        
        st.session_state.data_processor, st.session_state.train_loader = get_initialized_processor()




def prepare_input_grid(u_mag, aoa_deg, res=128):
    """
    Generates the 4-channel input for the FNO by rotating the flow vector.
    [u_inf, v_inf, mask, sdf]
    """
    # 1. Create the Grid (Airfoil stationary between x=0 and x=1)
    x = np.linspace(-0.5, 1.5, res)
    y = np.linspace(-1.0, 1.0, res)
    X, Y = np.meshgrid(x, y)
    
    # 2. Generate NACA 0012 Coordinates (No Rotation)
    t = 0.12 
    xc = np.linspace(0, 1, 200)
    yc = 5 * t * (0.2969*np.sqrt(xc) - 0.1260*xc - 0.3516*xc**2 + 0.2843*xc**3 - 0.1015*xc**4)
    
    # Create closed loop for Path and SDF
    pts = np.vstack([np.concatenate([xc, xc[::-1]]), 
                     np.concatenate([yc, -yc[::-1]])]).T
    
    # 3. Calculate Rotated Velocity Components
    theta = np.radians(aoa_deg)
    u_inf = u_mag * np.cos(theta)
    v_inf = u_mag * np.sin(theta)
    
    # 4. Compute Mask and SDF (Airfoil is stationary)
    path = mplPath(pts)
    grid_pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    
    is_inside = path.contains_points(grid_pts).reshape(res, res)
    mask = is_inside.astype(np.float32) 
    
    # 5. Compute Distance to BOUNDARY
    # Use scipy's cdist or a manual vectorized norm for speed on your 4090
    from scipy.spatial.distance import cdist
    # Distance from every grid point to every point on the airfoil perimeter
    dists = cdist(grid_pts, pts) 
    min_dists = np.min(dists, axis=1).reshape(res, res)

    # Apply AirFRANS SDF convention:
    # Points inside the solid airfoil: clipped to 0.0
    # Points outside (fluid domain): negative distance
    # Zero at the boundary
    sdf = np.where(is_inside, 0.0, -min_dists)

    # 6. Optional: Global Clipping following AirFRANS convention
    # AirFRANS clips SDF with max = 0 (no positive values allowed)
    sdf = np.clip(sdf, -3.0, 0.0) # Clip to AirFRANS convention: max = 0

    print(f"Prepared input grid with u_inf={u_inf:.2f}, v_inf={v_inf:.2f}, aoa={aoa_deg} deg, res={res}")

    print(f" SDF stats: min={sdf.min():.4f}, max={sdf.max():.4f}")
    
    # 7. Build the 4-Channel Tensor
    u_channel = np.full((res, res), u_inf)
    v_channel = np.full((res, res), v_inf)
    
    input_stack = np.stack([u_channel, v_channel, mask, sdf], axis=0)
    return torch.tensor(input_stack, dtype=torch.float32).unsqueeze(0)

def run_app_inference(model, x_input, res):

    dummy_y = torch.zeros((1, 4, res, res))
    # 2. Pack into the dictionary expected by DataProcessor
    sample = {'x': x_input, 'y': dummy_y} 
    
    with torch.no_grad():
        # Preprocess (Normalization)
        sample = st.session_state.data_processor.preprocess(sample)
        
        # Model Prediction (FNO works in Normalized space)
        y_norm_pred = model(sample['x'])
        
        # Postprocess (Back to Physical Units)
        y_phys_pred, _ = st.session_state.data_processor.postprocess(y_norm_pred, sample)
        
    return x_input.squeeze(0).cpu().numpy(), y_phys_pred.squeeze(0).cpu().numpy()



@st.cache_resource
def load_operator_dynamically(checkpoint_dir):
    # 1. Load the Metadata
    metadata_path = f"{checkpoint_dir}/model_metadata.pkl"
    # torch.load handles the "persistent id" error internally
    config = torch.load(metadata_path, map_location='cpu')
    
    # 2. Reconstruct the Model using saved config
    # This ensures n_modes, hidden_channels, etc. match perfectly
    model = FNO(
        n_modes=config.get('n_modes', (24, 24)),
        hidden_channels=config.get('hidden_channels', 128),
        in_channels=config.get('in_channels', 4),
        out_channels=config.get('out_channels', 3),
        factorization=config.get('factorization', 'tucker')
    )
    
    # 3. Load the Weights
    weights_path = f"{checkpoint_dir}/model_state_dict.pt"
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    model.eval()
    return model, config


# 1. Page Configuration
st.set_page_config(layout="wide", page_title="FNO Airfoil Inference")
st.title("🌬️ FNO Zero-Shot Airfoil Predictor")

@st.cache_resource
def load_operator(model_path):
    # Load your specific 128-channel FNO here
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    return model

# 2. Sidebar Controls
with st.sidebar:
    st.header("Simulation Parameters")
    u_mag = st.slider("Free-stream Velocity Magnitude (u_mag)", 10.0, 120.0, 100.0)
    aoa = st.number_input("Angle of Attack (deg)", -5.0, 20.0, 0.0)


    if u_mag != st.session_state.get('last_u_mag', None) or aoa != st.session_state.get('last_aoa', None):
        st.session_state.active_x = None
        st.session_state.active_y_truth = None
        st.session_state.last_u_mag = u_mag
        st.session_state.last_aoa = aoa

    model_res = st.selectbox("Inference Resolution", [128, 256,512,1028], index=0)

    # Let user pick an index
    max_idx = len(st.session_state.train_loader.dataset) - 1
    st.header("Load Test Sample from Dataset")
    selected_idx = st.number_input("Select Test Sample Index", 0, max_idx, 0)
    if st.button("Load from Dataset"):
        # Grab the sample and store it in session state
        # This will contain {'x': tensor, 'y': tensor}
        dataset = st.session_state.train_loader.dataset
        sample = dataset[selected_idx]
        
        # We add a batch dimension [1, C, H, W] for the FNO
        st.session_state.active_x = sample['x'].unsqueeze(0)
        st.session_state.active_y_truth = sample['y'].unsqueeze(0)
        st.success(f"Loaded Airfoil Sample #{selected_idx}")

if st.button("Run Inference"):

        # Load the saved "Training Truth"
    torch.serialization.add_safe_globals([zencfg.bunch.Bunch, torch._C._nn.gelu, neuralop.layers.spectral_convolution.SpectralConv])
    check_dir ="/home/timm/Projects/PIML/neuraloperator/tims/airfrans_all_out/checkpoints-all-weighted-L2"
    model,model_config = load_operator_dynamically(check_dir)

    with torch.no_grad():
        # Construct input tensor [1, 4, res, res]
        # Ch0: u_inf, Ch1: v_inf, Ch2: Mask, Ch3: SDF
        # For a demo, you can load a static 'NACA0012' SDF and rotate it by aoa
            # Priority: If a dataset sample is loaded, use it; otherwise, generate NACA
        if "active_x" in st.session_state:
            x_input = st.session_state.active_x
            source = "Dataset Sample"
        else:
            # Your existing prepare_input_grid function
            x_input = prepare_input_grid(u_mag, aoa, model_res)
            source = "NACA Generator"
            
        st.info(f"Showing results for {source}")
        x_raw, y_phys = run_app_inference(model, x_input, model_res)
        
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        col5, col6 = st.columns(2)

        with col1:
            st.subheader("Mask")
            fig, ax = plt.subplots()
            im = ax.imshow(x_raw[2], origin='lower', cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(im)
            st.pyplot(fig)
        with col2:
            st.subheader("SDF")
            fig, ax = plt.subplots()
            im = ax.imshow(x_raw[3], origin='lower', cmap='viridis', vmin=-1, vmax=0)
            plt.colorbar(im)
            st.pyplot(fig)

        with col3:
            st.subheader("U-Velocity (Ch 0)")
            fig, ax = plt.subplots()
            im = ax.imshow(y_phys[0], origin='lower', cmap='viridis', vmin=0, vmax=120)
            plt.colorbar(im)
            st.pyplot(fig)

        with col4:
            st.subheader("V-Velocity (Ch 1)")
            fig, ax = plt.subplots()
            im = ax.imshow(y_phys[1], origin='lower', cmap='RdBu_r', vmin=-5, vmax=5)
            plt.colorbar(im)
            st.pyplot(fig)

        with col5:
            st.subheader("Pressure (Cp) (Ch 2)")
            fig, ax = plt.subplots()
            im = ax.imshow(y_phys[2], origin='lower', cmap='plasma', vmin=-1, vmax=1)
            plt.colorbar(im)
            st.pyplot(fig)

        with col6 :
            st.subheader("Turbulent Viscosity (Nut) (Ch 3)")
            fig, ax = plt.subplots()
            # Nut is usually strictly positive and localized to the wake
            im = ax.imshow(y_phys[3], origin='lower', cmap='magma', vmin=0, vmax=1e-4)
            plt.colorbar(im)
            st.pyplot(fig)


