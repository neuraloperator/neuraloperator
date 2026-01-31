"""
Quick evaluation script to compute test metrics for trained FNO model.
"""

import torch
import numpy as np
from pathlib import Path
import argparse

from tims.airfrans.airfrans_dataset import load_airfrans_dataset
from neuralop.models import get_model
from airfrans_config import Default
from neuralop.losses import LpLoss, H1Loss


def evaluate_model(model, data_loader, data_processor, device):
    """Evaluate model on test set."""
    
    model = model.to(device)
    if data_processor is not None:
        data_processor = data_processor.to(device)
    
    # Loss functions
    l2_loss = LpLoss(d=2, p=2)
    h1_loss = H1Loss(d=2)
    
    model.eval()
    total_l2 = 0.0
    total_h1 = 0.0
    num_samples = 0
    
    # Per-channel metrics
    channel_names = ['u_velocity', 'v_velocity', 'pressure', 'nut_turbulence']
    channel_mse = [0.0] * 4
    channel_mae = [0.0] * 4
    
    with torch.no_grad():
        for batch in data_loader:
            # Move to device
            for key in batch:
                batch[key] = batch[key].to(device)
            
            batch_size = batch['x'].shape[0]
            
            # Preprocess
            if data_processor is not None:
                processed_batch = data_processor.preprocess(batch, batched=True)
            else:
                processed_batch = batch
            
            # Prediction
            y_pred = model(processed_batch['x'])
            
            # Postprocess
            if data_processor is not None:
                y_pred, sample = data_processor.postprocess(y_pred, batch, batched=True)
                y_true = sample['y']
            else:
                y_true = batch['y']
            
            # Overall losses
            l2_batch = l2_loss(y_pred, y_true).item()
            h1_batch = h1_loss(y_pred, y_true).item()
            
            total_l2 += l2_batch * batch_size
            total_h1 += h1_batch * batch_size
            num_samples += batch_size
            
            # Per-channel metrics
            for c in range(4):
                pred_c = y_pred[:, c:c+1]  # Keep channel dimension
                true_c = y_true[:, c:c+1]
                
                mse_c = torch.mean((pred_c - true_c)**2).item()
                mae_c = torch.mean(torch.abs(pred_c - true_c)).item()
                
                channel_mse[c] += mse_c * batch_size
                channel_mae[c] += mae_c * batch_size
    
    # Average metrics
    avg_l2 = total_l2 / num_samples
    avg_h1 = total_h1 / num_samples
    
    avg_channel_mse = [mse / num_samples for mse in channel_mse]
    avg_channel_mae = [mae / num_samples for mae in channel_mae]
    
    return {
        'l2_loss': avg_l2,
        'h1_loss': avg_h1,
        'channel_mse': dict(zip(channel_names, avg_channel_mse)),
        'channel_mae': dict(zip(channel_names, avg_channel_mae)),
        'num_samples': num_samples
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate FNO airfoil model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str,
                       default='/home/timm/Projects/PIML/neuraloperator/tims/airfrans/consolidated_data',
                       help='Root directory of dataset')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Default().to_dict()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    train_loader, test_loaders, data_processor = load_airfrans_dataset(
        train_split='scarce_train',
        test_splits=['full_test', 'aoa_test'],
        batch_size=4,  # Larger batch for evaluation
        test_batch_sizes=[4, 4],
        data_root=args.data_root,
        train_resolution=64,
        test_resolutions=[64, 128],
        encode_input=True,
        encode_output=False,  # Fixed to match corrected training
        encoding="channel-wise",
    )
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = get_model(config)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    # Evaluate on different test sets
    for resolution, test_loader in test_loaders.items():
        print(f"\n=== Evaluating on {resolution}x{resolution} test set ===")
        
        metrics = evaluate_model(model, test_loader, data_processor, device)
        
        print(f"Number of test samples: {metrics['num_samples']}")
        print(f"L2 Loss: {metrics['l2_loss']:.6f}")
        print(f"H1 Loss: {metrics['h1_loss']:.6f}")
        
        print("\nPer-channel metrics:")
        for channel, mse in metrics['channel_mse'].items():
            mae = metrics['channel_mae'][channel]
            rmse = np.sqrt(mse)
            print(f"  {channel:15} - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")


if __name__ == "__main__":
    main()