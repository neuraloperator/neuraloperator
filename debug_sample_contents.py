#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/timm/Projects/PIML/neuraloperator')

import torch
from torch.utils.data import DataLoader
from tims.airfransX5Y4.airfrans_trainerX5Y4_v1 import AirfransTrainer
from tims.losses.meta_losses import WeightedFieldwiseAggregatorLoss
from neuralop.losses import LpLoss
from neuralop.models import FNO
from tims.airfransX5Y4.airfrans_data_processor import AirfransDataProcessor

def quick_sample_debug():
    """Quick test to see sample contents during training and evaluation"""
    
    print("=== QUICK SAMPLE DEBUG TEST ===")
    
    # Create minimal model and trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a simple FNO model
    model = FNO(n_modes=(16, 16), in_channels=5, out_channels=4, hidden_channels=32, n_layers=4)
    
    # Create data processor
    data_processor = AirfransDataProcessor(
        resolution=128,
        grid_boundaries=[[0, 1], [0, 1]],
        device=device
    )
    
    # Create the loss function
    fieldwise_loss = WeightedFieldwiseAggregatorLoss(
        losses={
            'u_def': LpLoss(d=2, p=2, reduction='sum'),
            'v_def': LpLoss(d=2, p=2, reduction='sum'),
            'cp': LpLoss(d=2, p=2, reduction='sum'),
            'lognutratio': LpLoss(d=2, p=2, reduction='sum'),
        },
        weights={'u_def': 1.0, 'v_def': 1.0, 'cp': 1.0, 'lognutratio': 1.0}
    ).to(device)
    
    # Create trainer
    trainer = AirfransTrainer(
        model=model,
        n_epochs=1,
        device=device,
        data_processor=data_processor,
        verbose=True
    )
    
    # Create fake data that matches expected format
    batch_size = 2
    x_raw = torch.randn(batch_size, 5, 128, 128).to(device)  # 5 input channels
    y_raw = torch.randn(batch_size, 4, 128, 128).to(device)  # 4 output channels
    
    fake_sample = {
        'x': x_raw,
        'y': y_raw,
        'props': {'some_prop': 'value'}  # Non-tensor property
    }
    
    eval_losses = {'weightedField': fieldwise_loss}
    
    print("\\n" + "="*60)
    print("TESTING EVALUATION SAMPLE PROCESSING")
    print("="*60)
    
    # Test evaluation batch processing
    try:
        eval_step_losses, _ = trainer.eval_one_batch(fake_sample, eval_losses, return_output=False)
        print(f"\\nEval result: {eval_step_losses}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\\n" + "="*60)
    print("TESTING TRAINING SAMPLE PROCESSING") 
    print("="*60)
    
    # Test training batch processing 
    try:
        trainer.epoch = 0  # Set epoch for debug printing
        loss, channel_losses = trainer.train_one_batch(0, fake_sample, fieldwise_loss)
        print(f"\\nTraining loss: {loss.item()}")
        print(f"Channel losses: {channel_losses}")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_sample_debug()