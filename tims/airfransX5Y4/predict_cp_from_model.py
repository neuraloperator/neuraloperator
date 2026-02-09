import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def extract_surface_cp(y_decoded, foil_coords, grid_bounds):
    """
    y_decoded: Tensor [1, 4, H, W] (Decoded Physical Units)
    foil_coords: Tensor [N, 2] (Physical X, Y coordinates of the foil)
    grid_bounds: List [xmin, xmax, ymin, ymax]
    """
    xmin, xmax, ymin, ymax = grid_bounds
    
    # 1. Normalize foil coordinates to the [-1, 1] range required by grid_sample
    # This maps the physical space to the FNO's canonical grid space
    x_norm = 2.0 * (foil_coords[:, 0] - xmin) / (xmax - xmin) - 1.0
    y_norm = 2.0 * (foil_coords[:, 1] - ymin) / (ymax - ymin) - 1.0
    
    # 2. Shape for grid_sample: [Batch, N_points, 1 (H), 2 (XY)]
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).unsqueeze(2)
    
    # 3. Extract the Cp channel (Index 2 in your Y tensor: [u, v, cp, nut])
    cp_field = y_decoded[:, 2:3, :, :]
    
    # 4. Interpolate
    # align_corners=True is critical to match the FNO grid boundaries
    surface_cp = F.grid_sample(cp_field, grid, mode='bilinear', align_corners=True)
    
    return surface_cp.flatten() # Returns 1D array of Cp along the surface




if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tims.airfransX5Y4.airfrans_datasetX5Y4_v1 import AirfransDataset
    from tims.airfransX5Y4.config_AirfransX5Y4_v1 import AirfransDatasetConfig,Default
    from tims.airfransX5Y4.airfrans_datasetX5Y4_v1 import load_airfrans_dataset
    from zencfg import make_config_from_cli


    archive_dir = "/home/timm/Projects/PIML/Dataset_PT_FNO/Archive"    
    index = 8


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