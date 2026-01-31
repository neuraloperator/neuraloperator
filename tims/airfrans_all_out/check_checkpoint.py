import torch
from neuralop.training import setup, AdamW


from tims.airfrans_all_out.airfrans_dataset_all import load_airfrans_dataset
from tims.airfrans_all_out.airfrans_trainer import AirfransAllTrainer
from tims.airfrans_all_out.config_airfrans_all_weightedLoss import Default
from zencfg import make_config_from_cli 
import zencfg.bunch
from neuralop import H1Loss, LpLoss, Trainer, get_model
from pathlib import Path
import neuralop.layers.spectral_convolution
import pickle
import sys
import os
import wandb
from tims.airfrans_all_out.config_airfrans_all_weightedLoss import Default

sys.path.insert(0, "../")


def calculate_u_mean(train_loader):
    total_sum = 0
    total_elements = 0
    print("Calculating dataset-wide U-mean...")
    for sample in train_loader:
        # Assuming sample['y'] is [Batch, 4, 128, 128] and u is index 0
        u_channel = sample['y'][:, 0, ...]
        total_sum += u_channel.sum().item()
        total_elements += u_channel.numel()
    
    u_mean = total_sum / total_elements
    print(f"Calculated Dataset U-mean: {u_mean:.4f}")
    return u_mean

def inspect_metadata(metadata_path):
    metadata_path = Path(metadata_path) / "model_metadata.pkl"
    
    if not metadata_path.exists():
        print(f"❌ Error: {metadata_path} not found.")
        return

    try:
        # Use torch.load instead of pickle.load to handle persistent IDs
        # map_location='cpu' ensures it doesn't try to jump to a GPU unexpectedly
        metadata = torch.load(metadata_path, map_location='cpu', weights_only=False)
        
        print(f"--- 📜 Metadata Audit: {metadata_path.name} ---")
        
        # In NeuralOperator, the metadata is often a dictionary 
        # but check if it's a nested 'config' object
        if hasattr(metadata, 'to_dict'):
            metadata = metadata.to_dict()

        # Display the core architectural settings
        for key, value in metadata.items():
            # Filter out giant internal states to keep the output readable
            if not isinstance(value, (dict, list, torch.Tensor)) or key in ['n_modes']:
                print(f"{key:<20}: {value}")
            elif isinstance(value, torch.Tensor):
                print(f"{key:<20}: Tensor of shape {list(value.shape)}")

    except Exception as e:
        print(f"❌ Failed to load metadata: {e}")

def inspect_checkpoint_mismatch(model, checkpoint_dir):
    # 1. Define the specific weight file path
    # The library trainer saves weights as 'model_state_dict.pt'
    torch.serialization.add_safe_globals([zencfg.bunch.Bunch, torch._C._nn.gelu, neuralop.layers.spectral_convolution.SpectralConv])


    checkpoint_path = os.path.join(checkpoint_dir, "model_state_dict.pt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find weights at {checkpoint_path}")

    # 2. Load the state dict
    # Weights_only=False is often needed if custom activations like GELU were used
    checkpoint_state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Look for keys starting with 'data_processor'
    processor_keys = [k for k in checkpoint_state.keys() if 'data_processor' in k or 'normalizer' in k]

    if processor_keys:
        print(f"✅ DataProcessor stats found in checkpoint! ({len(processor_keys)} keys)")
    else:
        print("❌ DataProcessor NOT in checkpoint. It must be loaded/refitted separately.")

    # 3. Defensive check: Ensure we didn't get a None or a full model
    if checkpoint_state is None:
        print("❌ Error: Loaded object is None. Check if the file is corrupted.")
        return
    
    # If the file contains the model object directly instead of a dict
    if not isinstance(checkpoint_state, dict):
        print("⚠️ Checkpoint is a Model instance, not a state_dict. Extracting...")
        checkpoint_state = checkpoint_state.state_dict()
    model_state = model.state_dict()
    
    checkpoint_keys = set(checkpoint_state.keys())
    model_keys = set(model_state.keys())

    # 2. Identify missing or unexpected keys
    missing = model_keys - checkpoint_keys
    unexpected = checkpoint_keys - model_keys
    common = model_keys & checkpoint_keys

    print(f"--- Checkpoint Audit: {checkpoint_path} ---")
    print(f"Missing in Checkpoint: {len(missing)}")
    for k in sorted(list(missing))[:5]: print(f"  [!] {k}")
    
    print(f"\nUnexpected in Checkpoint: {len(unexpected)}")
    for k in sorted(list(unexpected))[:5]: print(f"  [?] {k}")

# 3. Identify shape mismatches in common layers
    shape_mismatch = []
    for k in common:
        model_val = model_state[k]
        ckpt_val = checkpoint_state[k]
        
        # Check if both are Tensors before comparing shapes
        if torch.is_tensor(model_val) and torch.is_tensor(ckpt_val):
            if model_val.shape != ckpt_val.shape:
                shape_mismatch.append((k, model_val.shape, ckpt_val.shape))
        
        # If one is a dict and the other isn't, or shapes can't be compared
        elif type(model_val) != type(ckpt_val):
            print(f"  [!] Type mismatch at {k}: Model {type(model_val)} vs Checkpoint {type(ckpt_val)}")

    print(f"\nShape Mismatches: {len(shape_mismatch)}")
    for k, m_shape, c_shape in shape_mismatch:
        print(f"  [X] {k}: Model {m_shape} vs Checkpoint {c_shape}")
    
    if not missing and not unexpected and not shape_mismatch:
        print("\n✅ PERFECT MATCH: Weights are ready for inference.")

if __name__ == "__main__":

    config = make_config_from_cli(Default)
    config = config.to_dict()
    device, is_logger = setup(config)
    print(f"Using device: {device}")




        # 3. Setup Model from Config
    model = get_model(config).to(device)

    checkpoint_dir = Path("/home/timm/Projects/PIML/neuraloperator/tims/airfrans_all_out/checkpoints-all-weighted-L2")
    inspect_metadata(checkpoint_dir)
    inspect_checkpoint_mismatch(model, checkpoint_dir=checkpoint_dir)


    # load the dataset to get data processor and test loaders
    train_loader, test_loaders, data_processor = load_airfrans_dataset(
        data_dir=config.data.data_dir,
        dataset_name=config.data.dataset_name,
        train_split=config.data.train_split,
        test_splits=config.data.test_splits,
        batch_size=config.data.batch_size,
        test_batch_sizes=config.data.test_batch_sizes,
        test_resolutions=config.data.test_resolutions,
        train_resolution=config.data.train_resolution,
        encode_input=config.data.encode_input,    
        encode_output=config.data.encode_output, 
        encoding=config.data.encoding,
        channel_dim=1,
    )

    if data_processor is not None:
        data_processor = data_processor.to(device)

    # import trainer to get the plotter
    trainer = AirfransAllTrainer(
        model=model,
        n_epochs=config.opt.n_epochs,
        data_processor=data_processor,
        device=device,
        mixed_precision=config.opt.mixed_precision,
        eval_interval=config.opt.eval_interval,
        log_output=is_logger,
        use_distributed=config.distributed.use_distributed,
        verbose=config.verbose,
        wandb_log=config.wandb.log,
    )

    save_dir = Path("/home/timm/Projects/PIML/neuraloperator/tims/airfrans_all_out/posts_check_plots")
    save_dir.mkdir(parents=True, exist_ok=True) 
    trainer.plot_diagnostic_grid(loader=train_loader, epoch=0,save_dir=save_dir , sample_idx=13, prefix="post_check_train")



    calculate_u_mean(train_loader)



