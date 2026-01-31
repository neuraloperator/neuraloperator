import torch
from torch.utils.data import DataLoader

def verify_encoding_stats(dataset_class, config):
    print(f"\n{'='*20} Encoding Verification {'='*20}")
    
    # Test Scenarios
    scenarios = [
        {"input": False, "output": False, "label": "RAW (Unencoded)"},
        {"input": True, "output": True, "label": "ENCODED (Standardized)"}
    ]

    for scene in scenarios:
        # Update config temporarily
        config.encode_input = scene["input"]
        config.encode_output = scene["output"]
        
        # Instantiate dataset
        ds = dataset_class(
            root_dir=config.folder,
            train_split=config.train_split,
            encode_input=config.encode_input,
            encode_output=config.encode_output,
            # Pass other required params from your config...
        )
        
        loader = DataLoader(ds, batch_size=10, shuffle=False)
        x, y = next(iter(loader))

        print(f"\n--- Scenario: {scene['label']} ---")
        
        # Check Inputs (x)
        print(f"{'Input (x)':<12} | Mean: {x.mean():>10.4f} | Std: {x.std():>10.4f} | Shape: {list(x.shape)}")
        
        # Check Outputs (y)
        print(f"{'Output (y)':<12} | Mean: {y.mean():>10.4f} | Std: {y.std():>10.4f} | Shape: {list(y.shape)}")

    print(f"{'='*63}\n")


if __name__ == "__main__":
    from tims.airfrans.airfrans_dataset import AirfransDataset
    from tims.airfrans.airfrans_config import AirfransDatasetConfig

    config = AirfransDatasetConfig()
    verify_encoding_stats(AirfransDataset, config)  