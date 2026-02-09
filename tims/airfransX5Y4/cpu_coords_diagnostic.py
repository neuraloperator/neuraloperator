import pathlib
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

import neuralop
from neuralop.models import FNO
from  pathlib import Path
from neuralop.models.base_model import get_model
from tims.airfransX5Y4.airfrans_datasetX5Y4_v1 import SelectiveDataProcessor
sys.path.insert(0, "../")
from zencfg import make_config_from_cli
from neuralop.models.base_model import get_model

from tims.airfransX5Y4.config_AirfransX5Y4_v1 import Default
# Import your specific trainer/dataset classes here
torch.serialization.add_safe_globals([torch._C._nn.gelu, neuralop.layers.spectral_convolution.SpectralConv])

def sort_airfoil_coords(coords, values):
    """Fixes the 'zigzag' by sorting points by polar angle around centroid."""
    cx, cy = coords.mean(dim=0)
    angles = torch.atan2(coords[:, 1] - cy, coords[:, 0] - cx)
    indices = torch.argsort(angles)
    return coords[indices], values[indices]

def standalone_diagnostic(model_path, processor_path, data_sample_path,config):
    # 1. Load on CPU
    device = torch.device('cpu')
    
    model = get_model(config)
        # Load model state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()


    # 3. Load Processor
    # Ensure this is the DataProcessor class from your airfrans_datasetX5Y4_v1.py
    data_processor_dict = torch.load(processor_path, map_location=device)
    data_processor = SelectiveDataProcessor(config['data_processor'])
    data_processor.load_state_dict(data_processor_dict)
    data_processor.to(device)
    data_processor.eval()

    # 4. Load a specific sample (e.g., from your saved .pt archive)
    # [2026-02-02] Archive structure: {'x': x_data, 'y_raw': y_raw, 'props': {...}}
    sample = torch.load(data_sample_path, map_location=device)
    x = sample['x'].unsqueeze(0) # [1, 5, H, W]
    y_target = sample['y_raw'].unsqueeze(0)

    # 5. Prediction Pipeline
    with torch.no_grad():
        # Encode -> Predict -> Decode
        prep = data_processor.preprocess({'x': x, 'y': y_target})
        out_norm = model(prep['x'])
        decoded_pred, _ = data_processor.postprocess(out_norm, prep)
        
    # 6. Surface Extraction & Sorting
    y_phys = decoded_pred['y'].squeeze(0)
    props = sample['props']
    foil_coords = props['v_inf'] # Or wherever you stored the N points
    
    # Sort to fix zigzag
    cp_surf = y_phys[2] # Example: Cp is channel 2
    # (Assuming you have your interpolation logic here to get values AT foil_coords)
    sorted_coords, sorted_cp = sort_airfoil_coords(foil_coords, cp_surf)

    # 7. Final Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_coords[:, 0], -sorted_cp, 'r-', label='FNO Prediction')
    plt.invert_yaxis()
    plt.title(f"Diagnostic AoA: {props['aoa_deg']} | CL: ...")
    plt.show()

if __name__ == "__main__":

    MODEL_PATH = '/home/timm/Projects/PIML/neuraloperator/tims/airfransX5Y4/test_model'
    MODEL_STATE_DICT_PATH = Path(MODEL_PATH)/ 'model_state_dict.pt'
    DATA_PROCESSOR_PATH = Path(MODEL_PATH) / 'data_processor.pt'
    SAMPLE_DATA_PATH = Path('/home/timm/Projects/PIML/Dataset_PT_FNO_X5Y4/TrainingX5Y4_consolidated/airfoil_aoa_test_256x256.pt')


    # Example usage
    config = make_config_from_cli(Default)
    config = config.to_dict()
    standalone_diagnostic(MODEL_STATE_DICT_PATH, DATA_PROCESSOR_PATH, SAMPLE_DATA_PATH, config)