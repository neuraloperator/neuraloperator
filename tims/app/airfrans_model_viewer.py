import streamlit as st
import torch
import pickle
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

import zencfg

# Add the neuraloperator path to sys.path
NEURALOP_PATH = Path(__file__).parent.parent.parent
sys.path.append(str(NEURALOP_PATH))

import neuralop
from tims.airfransX5Y4.airfrans_datasetX5Y4_v1 import load_airfrans_dataset
from tims.airfransX5Y4.config_AirfransX5Y4_v1 import AirfransDatasetConfig, Default
from zencfg import make_config

torch.serialization.add_safe_globals([zencfg.bunch.Bunch, torch._C._nn.gelu, neuralop.layers.spectral_convolution.SpectralConv])

# Constants
DATASET_PATH = "/home/timm/Projects/PIML/Dataset_PT_FNO_X5Y4/TrainingX5Y4_consolidated"
MANIFEST_PATH = os.path.join(DATASET_PATH, "manifest.json")

def load_model_files(model_dir: str) -> tuple[Optional[Any], Optional[torch.nn.Module], Optional[Dict]]:
    """Load the three required files for model restoration."""
    data_processor_path = os.path.join(model_dir, "data_processor.pt")
    model_state_path = os.path.join(model_dir, "best_model_state_dict.pt")
    
    # Try both .pkl and .pt extensions for metadata
    model_metadata_pkl_path = os.path.join(model_dir, "best_model_metadata.pkl")
    model_metadata_pt_path = os.path.join(model_dir, "best_model_metadata.pt")
    
    data_processor = None
    model_state = None
    metadata = None
    
    try:
        if os.path.exists(data_processor_path):
            data_processor = torch.load(data_processor_path, map_location='cpu', weights_only=False)
            st.success(f"✓ Loaded data_processor.pt")
        else:
            st.error(f"❌ data_processor.pt not found")
            
        if os.path.exists(model_state_path):
            model_state = torch.load(model_state_path, map_location='cpu', weights_only=True)
            st.success(f"✓ Loaded best_model_state_dict.pt")
        else:
            st.error(f"❌ best_model_state_dict.pt not found")
            
        # Try loading metadata - both .pkl and .pt files should use torch.load()
        if os.path.exists(model_metadata_pt_path):
            try:
                metadata = torch.load(model_metadata_pt_path, map_location='cpu', weights_only=False)
                st.success(f"✓ Loaded best_model_metadata.pt")
            except Exception as e:
                st.error(f"Failed to load .pt metadata: {str(e)}")
                metadata = None
                
        elif os.path.exists(model_metadata_pkl_path):
            try:
                # Load .pkl file as if it was a .pt file (saved with torch.save)
                metadata = torch.load(model_metadata_pkl_path, map_location='cpu', weights_only=False)
                st.success(f"✓ Loaded best_model_metadata.pkl (using torch.load)")
            except Exception as e:
                st.error(f"Failed to load .pkl metadata with torch.load: {str(e)}")
                metadata = None
        else:
            st.error(f"❌ Neither best_model_metadata.pkl nor best_model_metadata.pt found")
            
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        
    return data_processor, model_state, metadata

def load_manifest() -> Dict[str, List[str]]:
    """Load the manifest.json file to get available splits."""
    try:
        with open(MANIFEST_PATH, 'r') as f:
            manifest = json.load(f)
        return manifest
    except Exception as e:
        st.error(f"Error loading manifest: {str(e)}")
        return {}

def create_dataset_config(split_names: List[str]) -> Dict:
    """Create dataset configuration."""
    config = make_config(Default).to_dict()
    
    # Update paths
    config['data']['data_dir'] = DATASET_PATH
    config['data']['train_split'] = split_names[0] if split_names else 'full_train'
    
    # Handle test splits - resolution is used in filename, not as prefix
    test_splits = split_names[1:] if len(split_names) > 1 else []
    
    # Use the configured resolutions from the default config
    # Resolution determines which file variant to load (e.g., 128x128, 256x256)
    config['data']['test_splits'] = test_splits
    config['data']['test_resolutions'] = config['data']['test_resolutions'][:len(test_splits)] if len(config['data']['test_resolutions']) >= len(test_splits) else [128] * len(test_splits)
    config['data']['test_batch_sizes'] = config['data']['test_batch_sizes'][:len(test_splits)] if len(config['data']['test_batch_sizes']) >= len(test_splits) else [64] * len(test_splits)
    
    return config

def plot_sample(loader, index: int = 0, device: str = 'cpu'):
    """Plot a sample from the data loader."""
    # Get the sample
    for i, sample in enumerate(loader):
        if i == index // loader.batch_size:
            idx_in_batch = index % loader.batch_size
            x = sample['x'][idx_in_batch].to(device)
            y = sample['y'][idx_in_batch].to(device)
            break
    else:
        st.error("Sample index out of range")
        return None
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # --- INPUTS ---
    u_inf, v_inf, re_val = x[0,0,0].item(), x[1,0,0].item(), x[4,0,0].item()
    
    axes[0, 0].imshow(x[2].cpu(), origin='lower') 
    axes[0, 0].set_title(f"Mask (1.0 inside)\nU_inf: {u_inf:.2f}")
    
    axes[0, 1].imshow(x[3].cpu(), origin='lower', cmap='seismic')
    axes[0, 1].set_title(f"SDF\nV_inf: {v_inf:.2f}")
    
    axes[0, 2].axis('off')
    axes[0, 2].text(0, 0.5, f"Reynolds Log: {re_val:.4f}\nIndex: {index}", fontsize=10)
    axes[0, 3].axis('off')
    
    # --- OUTPUTS with Shared Scaling for Velocity ---
    out_titles = ["U-Deficit", "V-Deficit", "Cp (Pressure)", "log(nut_ratio)"]
    
    # Calculate shared limits for U and V
    vel_min = min(y[0].min(), y[1].min()).cpu().item()
    vel_max = max(y[0].max(), y[1].max()).cpu().item()
    
    for i in range(4):
        cmap = 'viridis' if i < 2 else ('plasma' if i == 2 else 'magma')
        
        if i < 2:
            im = axes[1, i].imshow(y[i].cpu(), origin='lower', cmap=cmap, vmin=vel_min, vmax=vel_max)
            axes[1, i].set_title(f"{out_titles[i]}\n(Shared Scale)")
        else:
            im = axes[1, i].imshow(y[i].cpu(), origin='lower', cmap=cmap)
            axes[1, i].set_title(out_titles[i])
            
        fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.suptitle(f"AirFrans X5Y4 | Shared Velocity Scaling: [{vel_min:.2f}, {vel_max:.2f}]", fontsize=14)
    return fig

def plot_model_prediction(model, data_processor, loader, sample_index: int = 0, device: str = 'cpu'):
    """
    Make predictions with loaded model and visualize results.
    Based on the diagnostic plotting from the trainer.
    """
    if model is None:
        return None
    
    # Set to evaluation mode - only call eval() if it's an actual model object
    model.eval()
    has_data_processor = data_processor is not None and hasattr(data_processor, 'preprocess')
    if has_data_processor:
        if hasattr(data_processor, 'eval'):
            data_processor.eval()
    
    # Get the sample
    for i, batch in enumerate(loader):
        if i == sample_index // loader.batch_size:
            idx_in_batch = sample_index % loader.batch_size
            break
    else:
        return None
    
    with torch.no_grad():
        x_raw = batch['x'][idx_in_batch:idx_in_batch+1].to(device)
        y_raw = batch['y'][idx_in_batch:idx_in_batch+1].to(device)
        
        # Preprocess if data processor available
        if has_data_processor:
            sample = data_processor.preprocess({'x': x_raw, 'y': y_raw})
            x_input = sample['x'].to(device)
            y_norm_truth = sample['y'].to(device)
            
            # Make prediction
            y_norm_pred = model(x_input)
            
            # Postprocess prediction back to physical space
            try:
                y_decoded_dict, _ = data_processor.postprocess(y_norm_pred.clone(), sample)
                if isinstance(y_decoded_dict, dict):
                    y_decoded_pred = y_decoded_dict['y']
                else:
                    y_decoded_pred = y_decoded_dict
                    
                # Calculate residuals
                residual_norm = y_norm_truth - y_norm_pred
                residual = y_raw - y_decoded_pred
            except Exception as e:
                st.warning(f"Error in postprocessing: {e}. Using raw predictions.")
                y_decoded_pred = y_norm_pred
                residual = y_raw - y_decoded_pred
                residual_norm = residual
                
        else:
            # Direct prediction without preprocessing
            x_input = x_raw
            y_norm_pred = model(x_input)
            y_decoded_pred = y_norm_pred
            y_norm_truth = y_raw
            residual = y_raw - y_decoded_pred
            residual_norm = residual
        
        # Get resolution info
        resolution_h = x_raw.shape[-2]
        resolution_w = x_raw.shape[-1]
    
    # Create visualization
    n_out = y_raw.shape[1]
    fig, axes = plt.subplots(n_out, 5, figsize=(20, 4 * n_out))
    if n_out == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f"Model Prediction vs Truth | Resolution: {resolution_h}x{resolution_w} | Sample: {sample_index}", fontsize=16)
    
    n_labels = ['U-Deficit', 'V-Deficit', 'Cp (Pressure)', 'log(nut_ratio)']
    
    for i in range(n_out):
        # Extract scaling bounds
        y_min, y_max = y_raw[0, i].min().item(), y_raw[0, i].max().item()
        if has_data_processor:
            z_min, z_max = y_norm_truth[0, i].min().item(), y_norm_truth[0, i].max().item()
        else:
            z_min, z_max = y_min, y_max
        
        # Calculate stats
        res = residual[0, i].cpu().numpy()
        mae = np.abs(res).mean()
        mse = np.square(res).mean()
        
        stats_text = f"MAE: {mae:.4f}\nMSE: {mse:.4f}"
        
        # Column 1: Ground Truth
        im0 = axes[i, 0].imshow(y_raw[0, i].cpu(), origin='lower', vmin=y_min, vmax=y_max)
        axes[i, 0].set_title(f"Truth {n_labels[i] if i < len(n_labels) else f'Ch{i}'}\nRange: [{y_min:.2f}, {y_max:.2f}]", fontsize=9)
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Column 2: Truth Encoded (if available)
        if has_data_processor:
            im1 = axes[i, 1].imshow(y_norm_truth[0, i].cpu(), origin='lower', cmap='plasma', vmin=z_min, vmax=z_max)
            axes[i, 1].set_title(f"Truth Encoded\nRange: [{z_min:.2f}, {z_max:.2f}]", fontsize=9)
        else:
            im1 = axes[i, 1].imshow(y_raw[0, i].cpu(), origin='lower', vmin=y_min, vmax=y_max)
            axes[i, 1].set_title(f"Truth (No Encoding)", fontsize=9)
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Column 3: Prediction Encoded
        if has_data_processor:
            im2 = axes[i, 2].imshow(y_norm_pred[0, i].cpu(), origin='lower', cmap='plasma', vmin=z_min, vmax=z_max)
            axes[i, 2].set_title(f"Prediction Encoded", fontsize=9)
        else:
            im2 = axes[i, 2].imshow(y_norm_pred[0, i].cpu(), origin='lower', vmin=y_min, vmax=y_max)
            axes[i, 2].set_title(f"Raw Prediction", fontsize=9)
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # Column 4: Prediction Decoded
        im3 = axes[i, 3].imshow(y_decoded_pred[0, i].cpu(), origin='lower', vmin=y_min, vmax=y_max)
        axes[i, 3].set_title(f"Prediction {n_labels[i] if i < len(n_labels) else f'Ch{i}'}\n{stats_text}", fontsize=9)
        fig.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)
        
        # Column 5: Residual Error
        max_err = np.max(np.abs(res))
        im4 = axes[i, 4].imshow(res, origin='lower', cmap='RdBu_r', vmin=-max_err, vmax=max_err)
        axes[i, 4].set_title(f"Residual Error\nMax: ±{max_err:.4f}", fontsize=9)
        fig.colorbar(im4, ax=axes[i, 4], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig

def load_model_and_predict(model_state, data_processor_state, config):
    """
    Load model weights and create model for prediction.
    Also try to recreate data processor if possible.
    """
    if model_state is None:
        return None, None
    
    try:
        # Create model from config - need to import the model creation logic
        from neuralop.models import FNO
        
        # Extract model config (this is a simplified version)
        # You may need to adjust based on your actual model configuration
        model_config = config.get('model', {})
        
        # Create FNO model with basic config
        model = FNO(
            n_modes=model_config.get('n_modes', [24, 24]),
            hidden_channels=model_config.get('hidden_channels', 64),
            in_channels=model_config.get('data_channels', 5),
            out_channels=model_config.get('out_channels', 4),
            n_layers=model_config.get('n_layers', 4)
        )
        
        # Load state dict
        model.load_state_dict(model_state)
        model.eval()
        
        # Try to recreate data processor if possible
        data_processor = None
        if data_processor_state is not None:
            try:
                # Try to create a data processor from the loaded dataset
                # This is a fallback - we'll just return the state dict for now
                # and handle it gracefully in the prediction function
                if hasattr(data_processor_state, 'preprocess'):
                    data_processor = data_processor_state
                else:
                    st.warning("⚠️ Data processor loaded as state dict only. Predictions will be made without preprocessing.")
                    data_processor = None
            except Exception as e:
                st.warning(f"⚠️ Could not recreate data processor: {e}")
                data_processor = None
        
        return model, data_processor
        
    except Exception as e:
        st.error(f"Error creating model: {str(e)}")
        return None, None

def main():
    st.set_page_config(page_title="AirFrans Model Viewer", layout="wide")
    
    st.title("🌊 AirFrans Model & Dataset Viewer")
    st.markdown("Load trained models and explore the AirFrans dataset")
    
    # Sidebar for model loading
    st.sidebar.header("📁 Model Loading")
    
    # Model directory selection
    model_dir = st.sidebar.text_input(
        "Model Directory Path", 
        value="",
        help="Path to directory containing model files"
    )
    
    if model_dir and os.path.exists(model_dir):
        st.sidebar.success("✓ Directory exists")
        
        # Load model files
        with st.sidebar.expander("Load Model Files", expanded=True):
            if st.button("🔄 Load Model Files"):
                data_processor, model_state, metadata = load_model_files(model_dir)
                
                # Store in session state
                st.session_state.data_processor_state = data_processor  # This might be a state dict
                if metadata:
                    st.sidebar.subheader("📊 Model Info")
                    if isinstance(metadata, dict):
                        for key, value in metadata.items():
                            if key in ['epoch', 'train_loss', 'eval_loss', 'learning_rate']:
                                st.sidebar.metric(key.replace('_', ' ').title(), f"{value:.4f}" if isinstance(value, float) else str(value))
    
    elif model_dir:
        st.sidebar.error("❌ Directory does not exist")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🔧 Dataset Configuration")
        
        # Load manifest
        manifest = load_manifest()
        if manifest:
            st.success(f"✓ Found {len(manifest)} splits in manifest")
            
            # Split selection
            available_splits = list(manifest.keys())
            selected_splits = st.multiselect(
                "Select Dataset Splits",
                available_splits,
                default=['full_train'] if 'full_train' in available_splits else available_splits[:1],
                help="First split will be used as training set, others as test sets"
            )
            
            if selected_splits:
                st.info(f"Train split: **{selected_splits[0]}**")
                if len(selected_splits) > 1:
                    st.info(f"Test splits: **{', '.join(selected_splits[1:])}**")
                
                # Load dataset button
                if st.button("🔄 Load Dataset"):
                    try:
                        with st.spinner("Loading dataset..."):
                            config = create_dataset_config(selected_splits)
                            
                            # Display config info for debugging
                            st.info(f"Loading with config:")
                            st.json({
                                "data_dir": config['data']['data_dir'],
                                "train_split": config['data']['train_split'],
                                "test_splits": config['data']['test_splits'],
                                "test_resolutions": config['data']['test_resolutions'],
                                "test_batch_sizes": config['data']['test_batch_sizes']
                            })
                            
                            train_loader, test_loaders, data_processor_from_dataset = load_airfrans_dataset(
                                data_dir=config['data']['data_dir'],
                                dataset_name=config['data']['dataset_name'],
                                train_split=config['data']['train_split'],
                                test_splits=config['data']['test_splits'],
                                batch_size=config['data']['batch_size'],
                                test_batch_sizes=config['data']['test_batch_sizes'],
                                test_resolutions=config['data']['test_resolutions'],
                                encode_input=config['data']['encode_input'],    
                                encode_output=config['data']['encode_output'], 
                                encoding=config['data']['encoding'],
                                channel_dim=1,
                            )
                            
                            # Store in session state
                            st.session_state.train_loader = train_loader
                            st.session_state.test_loaders = test_loaders
                            st.session_state.config = config
                            st.session_state.selected_splits = selected_splits
                            
                            st.success("✓ Dataset loaded successfully!")
                            
                            # Display dataset info
                            st.subheader("📈 Dataset Statistics")
                            st.metric("Training Samples", len(train_loader.dataset))
                            
                            # Handle test_loaders - can be either list or dict
                            if isinstance(test_loaders, dict):
                                for i, (split_key, test_loader) in enumerate(test_loaders.items()):
                                    if hasattr(test_loader, 'dataset'):
                                        resolution = config['data']['test_resolutions'][i] if i < len(config['data']['test_resolutions']) else 128
                                        st.metric(f"{split_key} ({resolution}x{resolution})", len(test_loader.dataset))
                                    else:
                                        st.warning(f"Test loader '{split_key}' is not a valid DataLoader: {type(test_loader)}")
                            elif isinstance(test_loaders, list):
                                for i, test_loader in enumerate(test_loaders):
                                    if hasattr(test_loader, 'dataset'):
                                        split_name = selected_splits[i+1] if i+1 < len(selected_splits) else f"Test Set {i+1}"
                                        resolution = config['data']['test_resolutions'][i] if i < len(config['data']['test_resolutions']) else 128
                                        st.metric(f"{split_name} ({resolution}x{resolution})", len(test_loader.dataset))
                                    else:
                                        st.warning(f"Test loader {i+1} is not a valid DataLoader: {type(test_loader)}")
                            else:
                                st.warning(f"Expected test_loaders to be a list or dict, got {type(test_loaders)}")
                                
                    except Exception as e:
                        st.error(f"Error loading dataset: {str(e)}")
                        st.error(f"Exception type: {type(e).__name__}")
                        import traceback
                        st.error(f"Full traceback:")
                        st.code(traceback.format_exc())
    
    with col2:
        st.header("🔍 Dataset Visualization")
        
        # Check if dataset is loaded
        if 'train_loader' in st.session_state:
            # Loader selection - train_loader is single, test_loaders is a list
            loader_options = {"Training Set": st.session_state.train_loader}
            
            if 'test_loaders' in st.session_state:
                test_loaders = st.session_state.test_loaders
                if isinstance(test_loaders, dict):
                    for i, (split_key, test_loader) in enumerate(test_loaders.items()):
                        if hasattr(test_loader, 'dataset'):
                            resolution = st.session_state.config['data']['test_resolutions'][i] if i < len(st.session_state.config['data']['test_resolutions']) else 128
                            loader_options[f"{split_key} ({resolution}x{resolution})"] = test_loader
                elif isinstance(test_loaders, list):
                    for i, test_loader in enumerate(test_loaders):
                        if hasattr(test_loader, 'dataset'):
                            split_name = st.session_state.selected_splits[i+1] if i+1 < len(st.session_state.selected_splits) else f"Test Set {i+1}"
                            resolution = st.session_state.config['data']['test_resolutions'][i] if i < len(st.session_state.config['data']['test_resolutions']) else 128
                            loader_options[f"{split_name} ({resolution}x{resolution})"] = test_loader
            
            selected_loader_name = st.selectbox("Select Data Loader", list(loader_options.keys()))
            selected_loader = loader_options[selected_loader_name]
            
            # Sample index selection
            if hasattr(selected_loader, 'dataset'):
                max_samples = len(selected_loader.dataset)
                sample_index = st.number_input(
                    "Sample Index", 
                    min_value=0, 
                    max_value=max_samples-1, 
                    value=0,
                    help=f"Select sample from 0 to {max_samples-1}"
                )
                
                # Plot sample
                if st.button("🎨 Plot Sample"):
                    try:
                        fig = plot_sample(selected_loader, sample_index)
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)  # Clean up
                    except Exception as e:
                        st.error(f"Error plotting sample: {str(e)}")
                
                # Model prediction section
                st.markdown("---")
                st.subheader("🤖 Model Prediction")
                
                if 'model_state' in st.session_state and st.session_state.model_state is not None:
                    if st.button("🔮 Generate Prediction"):
                        try:
                            with st.spinner("Loading model and generating prediction..."):
                                # Load model
                                model, data_processor = load_model_and_predict(
                                    st.session_state.model_state,
                                    st.session_state.get('data_processor_state'),
                                    st.session_state.config
                                )
                                
                                if model is not None:
                                    # Generate prediction plot
                                    fig = plot_model_prediction(
                                        model, 
                                        data_processor, 
                                        selected_loader, 
                                        sample_index,
                                        device='cpu'
                                    )
                                    
                                    if fig:
                                        st.pyplot(fig)
                                        plt.close(fig)
                                    else:
                                        st.error("Failed to generate prediction plot")
                                else:
                                    st.error("Failed to load model")
                                    
                        except Exception as e:
                            st.error(f"Error generating prediction: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
                else:
                    st.info("💡 Load a model first to generate predictions")
            else:
                st.error("Selected loader is not a valid DataLoader object")
        else:
            st.info("👆 Load a dataset first to visualize samples")
    
    # Footer
    st.markdown("---")
    st.markdown("*AirFrans Model Viewer - Built with Streamlit*")

if __name__ == "__main__":
    main()