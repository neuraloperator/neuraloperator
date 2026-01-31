"""
Script to consolidate individual AirFRANS simulation files into PTDataset format.
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm


def consolidate_airfrans_dataset(
    data_root: Path,
    output_dir: Path,
    splits_config: dict,
    xlim: float = 2.0,
    ylim: float = 2.0,
    grid_sizes: list = [64, 128 , 256]
):
    """
    Consolidate individual simulation .pt files into PTDataset format.
    
    Args:
        data_root: Root directory containing simulation folders
        output_dir: Directory to save consolidated files
        splits_config: Dict mapping split names to their file lists from manifest
        xlim, ylim: Domain parameters used in filenames
        grid_sizes: List of grid resolutions to process
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for grid_size in grid_sizes:
        print(f"\nProcessing grid size {grid_size}x{grid_size}")
        
        for split_name, sim_dirs in splits_config.items():
            if not sim_dirs:  # Skip empty splits
                continue
                
            print(f"  Processing {split_name} split with {len(sim_dirs)} simulations")
            
            # Collect all data for this split
            all_x = []
            all_y = []
            
            for sim_dir in tqdm(sim_dirs, desc=f"Loading {split_name}"):
                sim_path = data_root / sim_dir
                data_file = sim_path / f"{sim_dir}_Cp_X{int(xlim)}_Y{int(ylim)}_G{grid_size}x{grid_size}.pt"
                
                if data_file.exists():
                    try:
                        data = torch.load(data_file)
                        all_x.append(data['x'].unsqueeze(0))  # Add batch dimension
                        all_y.append(data['y'].unsqueeze(0))  # Add batch dimension
                    except Exception as e:
                        print(f"Error loading {data_file}: {e}")
                        continue
                else:
                    print(f"Warning: {data_file} not found")
            
            if all_x:
                # Stack all samples along batch dimension
                consolidated_x = torch.cat(all_x, dim=0)  # Shape: [N, C, H, W]
                consolidated_y = torch.cat(all_y, dim=0)  # Shape: [N, C, H, W]
                
                consolidated_data = {
                    'x': consolidated_x,
                    'y': consolidated_y
                }
                
                # Determine if this is train or test split
                split_type = 'train' if 'train' in split_name else 'test'
                output_file = output_dir / f"airfoil_cp_{split_name}_{grid_size}.pt"
                
                torch.save(consolidated_data, output_file)
                print(f"  Saved {output_file} with {consolidated_x.shape[0]} samples")
                print(f"    Input shape: {consolidated_x.shape}")
                print(f"    Output shape: {consolidated_y.shape}")


def main():
    # Configuration
    data_root = Path("/home/timm/Projects/PIML/Dataset")
    output_dir = Path("/home/timm/Projects/PIML/neuraloperator/tims/airfrans/consolidated_Cp_data")
    manifest_file = data_root / "manifest.json"
    
    # Load manifest
    if manifest_file.exists():
        with open(manifest_file, 'r') as f:
            manifest_data = json.load(f)
    else:
        print(f"Error: Manifest file not found at {manifest_file}")
        return
    
    # Define which splits to use for train vs test
    # You can adjust this mapping based on your needs
    splits_config = {
        # Training splits - will be combined into one train file
        'full_train': manifest_data.get('full_train', []),
        'scarce_train': manifest_data.get('scarce_train', []),
        'aoa_train': manifest_data.get('aoa_train', []),
        'reynolds_train': manifest_data.get('reynolds_train', []),
        
        # Test splits - each will create a separate test file
        'full_test': manifest_data.get('full_test', []),
        'aoa_test': manifest_data.get('aoa_test', []),
        'reynolds_test': manifest_data.get('reynolds_test', [])
           
        }
    
    # Run consolidation
    consolidate_airfrans_dataset(
        data_root=data_root,
        output_dir=output_dir,
        splits_config=splits_config,
        xlim=2,
        ylim=2,
        grid_sizes=[64, 128, 256]
    )
    
    print(f"\nConsolidation complete! Files saved to {output_dir}")
    print("You can now use these with PTDataset:")
    for f in output_dir.glob("*.pt"):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()