"""
Training a neural operator on Darcy-Flow - Author Robert Joseph
========================================
In this example, we demonstrate how to use the small Darcy-Flow example we ship with the Incremental package
"""

import torch
import sys
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss

device = 'cpu'

def test_incremental_model_training(incremental_loss_gap, incremental, incremental_resolution,
                                    incremental_eps = 0.999, incremental_buffer=5,
                                    incremental_max_iter=1, incremental_grad_max_iter=10, 
                                    incremental_loss_eps=0.01, incremental_res_gap=100, dataset_resolution=16, dataset_sublist=[1], dataset_indices=[0]):
    """_summary_
    Args:
        incremental : bool, default is False
            if True, use the base incremental algorithm which is based on gradient variance
            uses the incremental_grad_eps parameter - set the threshold for gradient variance
            uses the incremental_buffer paramater - sets the number of buffer modes to calculate the gradient variance
            uses the incremental_max_iter parameter - sets the initial number of iterations
            uses the incremental_grad_max_iter parameter - sets the maximum number of iterations to accumulate the gradients
        incremental_loss_gap : bool, default is False
            if True, use the incremental algorithm based on loss gap
            uses the incremental_loss_eps parameter
        incremental_resolution : bool, default is False
            if True, increase the resolution of the input incrementally
            uses the incremental_res_gap parameter
            uses the dataset_sublist parameter - a list of resolutions to use
            uses the dataset_indices parameter - a list of indices of the dataset to slice to regularize the input resolution
            uses the dataset_resolution parameter - the resolution of the input
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
    model = FNO(
        n_modes=(16, 16),
        hidden_channels=32,
        projection_channels=64,
        incremental_n_modes=starting_modes)
    model = model.to(device)
    n_params = count_params(model)

    # Set up the optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=8e-3,
        weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Set up the losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    train_loss = h1loss
    eval_losses = {'h1': h1loss, 'l2': l2loss}
    print('\n### N PARAMS ###\n', n_params)
    print('\n### OPTIMIZER ###\n', optimizer)
    print('\n### SCHEDULER ###\n', scheduler)
    print('\n### LOSSES ###')
    print(f'\n * Train: {train_loss}')
    print(f'\n * Test: {eval_losses}')
    sys.stdout.flush()

    # Set up the trainer
    trainer = Trainer(
        model,
        n_epochs=20,
        device=device,
        mg_patching_levels=0,
        wandb_log=False,
        log_test_interval=3,
        use_distributed=False,
        verbose=True,
        incremental_loss_gap=incremental_loss_gap,
        incremental=incremental,
        incremental_resolution=incremental_resolution,
        incremental_eps=incremental_eps, incremental_buffer=incremental_buffer,
        incremental_max_iter=incremental_max_iter, 
        incremental_grad_max_iter=incremental_grad_max_iter, 
        incremental_loss_eps=incremental_loss_eps, 
        incremental_res_gap=incremental_res_gap, 
        dataset_resolution=dataset_resolution, dataset_sublist=dataset_sublist, 
        dataset_indices=dataset_indices)

    # Train the model
    trainer.train(
        train_loader,
        test_loaders,
        output_encoder,
        model,
        optimizer,
        scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses)

    if incremental_mode:
        # Check that the number of modes has dynamically increased (Atleast for
        # these settings on this dataset it should increase)
        assert model.fno_blocks.convs.incremental_n_modes > starting_modes

    if not baseline:
        # Check that the number of modes has not dynamically increased (Atleast
        # for these settings on this dataset it should increase)
        assert model.fno_blocks.convs.incremental_n_modes == starting_modes



# Test Baseline Model first
test_incremental_model_training(
    incremental_loss_gap=False,
    incremental=False,
    incremental_resolution=False)

# Test Incremental Loss Gap
test_incremental_model_training(
    incremental_loss_gap=True,
    incremental=False,
    incremental_resolution=False)

# Test Incremental
test_incremental_model_training(
    incremental_loss_gap=False,
    incremental=True,
    incremental_resolution=False)

# Test Incremental Resolution
test_incremental_model_training(
    incremental_loss_gap=False,
    incremental=False,
    incremental_resolution=True)

# Test Incremental + Incremental Resolution
test_incremental_model_training(
    incremental_loss_gap=False,
    incremental=True,
    incremental_resolution=True)

# Test Incremental Loss Gap + Incremental Resolution
test_incremental_model_training(
    incremental_loss_gap=True,
    incremental=False,
    incremental_resolution=True)