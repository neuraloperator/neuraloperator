"""
Training a neural operator on Darcy-Flow - Author Robert Joseph
========================================
In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package
"""

import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO, FNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss

device = 'cpu'

def test_incremental_model_training(incremental_loss_gap=False, incremental=False, incremental_resolution=False):
    """_summary_

    Args:
            incremental_loss_gap (bool, optional): Loss gap method. Defaults to False.
            incremental (bool, optional): Gradient explained method. Defaults to False.
            incremental_resolution (bool, optional): Increase the resolution dynamically. Defaults to False.
    """        
    # DATASET
    # Loading the Darcy flow dataset
    train_loader, test_loaders, output_encoder = load_darcy_flow_small(
            n_train=100, batch_size=16, 
            test_resolutions=[16, 32], n_tests=[100, 50],
            test_batch_sizes=[32, 32],
            )
    
    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    incremental_mode = incremental_loss_gap or incremental
    baseline = incremental_mode
    
    if incremental_loss_gap:
        print('### INCREMENTAL LOSS GAP ###')
    if incremental:
        print('### INCREMENTAL ###')
    if incremental_resolution:
        print('### INCREMENTAL RESOLUTION ###')
    if not baseline and not incremental_resolution:
        print('### BASELINE ###')
                    
    # Set up the incremental FNO model
    if incremental_mode:
        starting_modes = (2, 2)
    else:
        starting_modes = (16, 16)

    # set up model
    model = FNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, incremental_n_modes=starting_modes)
    model = model.to(device)
    n_params = count_params(model)

    # Set up the optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Set up the losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    train_loss = h1loss
    eval_losses={'h1': h1loss, 'l2': l2loss}
    print('\n### N PARAMS ###\n', n_params)
    print('\n### OPTIMIZER ###\n', optimizer)
    print('\n### SCHEDULER ###\n', scheduler)
    print('\n### LOSSES ###')
    print(f'\n * Train: {train_loss}')
    print(f'\n * Test: {eval_losses}')
    sys.stdout.flush()
    
    # Set up the trainer
    trainer = Trainer(model, n_epochs=20, device=device, mg_patching_levels=0, wandb_log=False, log_test_interval=3, use_distributed=False, verbose=True, incremental_loss_gap = incremental_loss_gap, incremental = incremental, incremental_resolution = incremental_resolution, dataset_name="SmallDarcy")
        
    # Train the model
    trainer.train(train_loader, test_loaders, output_encoder, model, optimizer, scheduler, regularizer=False, training_loss=train_loss, eval_losses=eval_losses)
    
    if incremental_mode:
        # Check that the number of modes has dynamically increased (Atleast for these settings on this dataset it should increase)
        assert model.convs.incremental_n_modes > starting_modes
    
    if not baseline:
        # Check that the number of modes has not dynamically increased (Atleast for these settings on this dataset it should increase)
        assert model.convs.incremental_n_modes == starting_modes
    
# Test Baseline Model first
test_incremental_model_training(incremental_loss_gap=False, incremental=False, incremental_resolution=False)

# Test Incremental Loss Gap
test_incremental_model_training(incremental_loss_gap=True, incremental=False, incremental_resolution=False)

# Test Incremental
test_incremental_model_training(incremental_loss_gap=False, incremental=True, incremental_resolution=False)

# Test Incremental Resolution
test_incremental_model_training(incremental_loss_gap=False, incremental=False, incremental_resolution=True)

# Test Incremental + Incremental Resolution
test_incremental_model_training(incremental_loss_gap=False, incremental=True, incremental_resolution=True)

# Test Incremental Loss Gap + Incremental Resolution
test_incremental_model_training(incremental_loss_gap=True, incremental=False, incremental_resolution=True)
