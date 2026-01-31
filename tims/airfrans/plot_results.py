"""
Script to visualize FNO predictions on AirFRANS dataset.
Shows input conditions, model predictions, ground truth, and error maps.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys

# Add the parent directories to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import your dataset and model utilities
from tims.airfrans.airfrans_dataset import load_airfrans_dataset
from neuralop.models import get_model
from config.models import FNO_Small2d
from tims.airfrans.airfrans_config import Default
from zencfg import make_config_from_cli


def load_trained_model(checkpoint_path, config):
    """Load trained model from checkpoint."""
    model = get_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint with {checkpoint.get('n_params', 'unknown')} parameters")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def load_model_from_wandb(run_id, project="neuraloperator-airfrans", entity="tim-mak-ntnu"):
    """Load model from WandB artifact."""
    import wandb
    
    # Initialize wandb
    run = wandb.init(project=project, entity=entity, job_type="inference")
    
    # Get the model artifact
    artifact = run.use_artifact(f'{project}/model-checkpoint:latest')
    artifact_dir = artifact.download()
    
    # Find the model checkpoint
    checkpoint_path = None
    for file in Path(artifact_dir).glob("*.pt"):
        if "final" in file.name:
            checkpoint_path = file
            break
    
    if checkpoint_path is None:
        raise FileNotFoundError("No checkpoint found in WandB artifact")
    
    print(f"Downloaded checkpoint from WandB: {checkpoint_path}")
    wandb.finish()
    
    return checkpoint_path


def plot_airfoil_results(model, data_loader, data_processor, device, num_samples=3, save_path=None):
    """
    Plot model predictions vs ground truth for airfoil flow fields.
    
    Args:
        model: Trained FNO model
        data_loader: Test data loader
        data_processor: Data processor for normalization
        device: Device to run inference on
        num_samples: Number of test samples to visualize
        save_path: Optional path to save plots
    """
    
    # Channel names for labeling
    input_channels = ['u_input', 'v_input', 'mask_binary', 'sdf_fixed']
    output_channels = ['u_target', 'v_target', 'p_target', 'nut_fixed']
    
    # Move model to device
    model = model.to(device)
    if data_processor is not None:
        data_processor = data_processor.to(device)
    
    with torch.no_grad():
        for sample_idx, batch in enumerate(data_loader):
            if sample_idx >= num_samples:
                break
                
            # Take first sample from batch
            sample = {k: v[0:1] for k, v in batch.items()}  # Keep batch dim but only first sample
            
            # Move to device
            for key in sample:
                sample[key] = sample[key].to(device)
            
            # Get data for visualization - ground truth is already in physical units
            x_raw = sample['x'].cpu().squeeze(0)  # Input data (already physical)
            y_true = sample['y'].cpu().squeeze(0)  # Ground truth (already physical)
            
            # Preprocess for model input
            if data_processor is not None:
                processed_sample = data_processor.preprocess(sample, batched=True)
            else:
                processed_sample = sample
            
            # Model prediction
            y_pred = model(processed_sample['x'])
            
            # Postprocess prediction
            if data_processor is not None:
                y_pred, _ = data_processor.postprocess(y_pred, sample, batched=True)
            
            y_pred = y_pred.cpu().squeeze(0)  # Remove batch dim
            
            # Create comprehensive plot
            fig = plt.figure(figsize=(20, 16))
            fig.suptitle(f'Airfoil Flow Prediction - Sample {sample_idx + 1}', fontsize=16, fontweight='bold')
            
            # Plot input conditions (4 channels)
            for i in range(4):
                ax = plt.subplot(4, 7, i + 1)
                if input_channels[i] == 'mask_binary':
                    im = ax.imshow(x_raw[i].numpy(), cmap='gray', aspect='equal', vmin=0, vmax=1)
                    ax.set_title(f'Input: {input_channels[i]}')
                elif input_channels[i] == 'sdf_fixed':
                    im = ax.imshow(x_raw[i].numpy(), cmap='RdBu_r', aspect='equal')
                    ax.set_title(f'Input: {input_channels[i]}')
                else:  # velocity inputs
                    im = ax.imshow(x_raw[i].numpy(), cmap='RdBu_r', aspect='equal')
                    ax.set_title(f'Input: {input_channels[i]}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Plot outputs: Ground Truth, Prediction, Error
            for i in range(4):
                row = i
                
                # Ground Truth
                ax = plt.subplot(4, 7, row * 7 + 4)
                y_true_np = y_true[i].numpy()
                im = ax.imshow(y_true_np, cmap='RdBu_r', aspect='equal')
                ax.set_title(f'True {output_channels[i]}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Prediction
                ax = plt.subplot(4, 7, row * 7 + 5)
                y_pred_np = y_pred[i].numpy()
                im = ax.imshow(y_pred_np, cmap='RdBu_r', aspect='equal')
                ax.set_title(f'Pred {output_channels[i]}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Error Map
                ax = plt.subplot(4, 7, row * 7 + 6)
                error = np.abs(y_true_np - y_pred_np)
                im = ax.imshow(error, cmap='Reds', aspect='equal')
                ax.set_title(f'|Error| {output_channels[i]}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Statistics
                ax = plt.subplot(4, 7, row * 7 + 7)
                ax.axis('off')
                
                # Compute metrics
                mse = np.mean(error**2)
                mae = np.mean(error)
                max_err = np.max(error)
                rel_err = np.mean(error) / (np.mean(np.abs(y_true_np)) + 1e-8) * 100
                
                stats_text = f'{output_channels[i]}\n'
                stats_text += f'MSE: {mse:.4f}\n'
                stats_text += f'MAE: {mae:.4f}\n'
                stats_text += f'Max Error: {max_err:.4f}\n'
                stats_text += f'Rel Error: {rel_err:.2f}%\n'
                stats_text += f'True Range: [{y_true_np.min():.3f}, {y_true_np.max():.3f}]\n'
                stats_text += f'Pred Range: [{y_pred_np.min():.3f}, {y_pred_np.max():.3f}]'
                
                ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            if save_path:
                save_file = f"{save_path}_sample_{sample_idx + 1}.png"
                plt.savefig(save_file, dpi=150, bbox_inches='tight')
                print(f"Saved plot to {save_file}")
            
            plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize FNO airfoil predictions')
    parser.add_argument('--checkpoint', type=str, 
                       help='Path to model checkpoint')
    parser.add_argument('--wandb_run', type=str,
                       help='WandB run ID to download checkpoint from')
    parser.add_argument('--wandb_project', type=str, default='neuraloperator-airfrans',
                       help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='tim-mak-ntnu',
                       help='WandB entity name')
    parser.add_argument('--data_root', type=str, 
                       default='/home/timm/Projects/PIML/neuraloperator/tims/airfrans/consolidated_data',
                       help='Root directory of dataset')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of test samples to visualize')
    parser.add_argument('--save_path', type=str, 
                       help='Base path to save plots (optional)')
    parser.add_argument('--resolution', type=int, default=64,
                       help='Test resolution to use')
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    checkpoint_path = args.checkpoint
    if checkpoint_path == "none":
        # Use random weights for testing
        checkpoint_path = None
        print("Using random weights for testing visualization")
    elif checkpoint_path is None:
        if args.wandb_run:
            checkpoint_path = load_model_from_wandb(args.wandb_run, args.wandb_project, args.wandb_entity)
        else:
            # Look for local checkpoint
            local_checkpoint = Path("./checkpoints/airfrans_fno_final.pt")
            if local_checkpoint.exists():
                checkpoint_path = local_checkpoint
                print(f"Using local checkpoint: {checkpoint_path}")
            else:
                print("No checkpoint specified. Use --checkpoint or --wandb_run")
                return
    
    # Load configuration (same as training)
    config = Default().to_dict()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    train_loader, test_loaders, data_processor = load_airfrans_dataset(
        train_split='scarce_train',
        test_splits=['full_test', 'aoa_test'],
        batch_size=1,  # Use batch size 1 for visualization
        test_batch_sizes=[1, 1],
        data_root=args.data_root,
        train_resolution=64,
        test_resolutions=[64, 128],
        encode_input=True,
        encode_output=False,  # Fixed to match corrected training
        encoding="channel-wise",
    )
    
    # Load trained model or create random model
    if checkpoint_path:
        print(f"Loading model from {checkpoint_path}")
        model = load_trained_model(checkpoint_path, config)
    else:
        print("Creating model with random weights")
        model = get_model(config)
        model.eval()
    
    # Select test loader based on resolution
    if args.resolution in test_loaders:
        test_loader = test_loaders[args.resolution]
        print(f"Using test data at resolution {args.resolution}x{args.resolution}")
    else:
        # Use first available test loader
        test_loader = next(iter(test_loaders.values()))
        print(f"Resolution {args.resolution} not available, using default test loader")
    
    # Create visualizations
    print(f"Creating visualizations for {args.num_samples} samples...")
    plot_airfoil_results(
        model=model,
        data_loader=test_loader,
        data_processor=data_processor,
        device=device,
        num_samples=args.num_samples,
        save_path=args.save_path
    )
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()