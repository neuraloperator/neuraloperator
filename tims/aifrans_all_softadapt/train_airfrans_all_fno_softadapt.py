"""
Training script for Airfrans flow  using neural operators.

This script trains a neural operator on the 2D Airfrans flow equation,
which models fluid flow through porous media. The script supports
distributed training and multi-grid patching for high-resolution data.
"""

import csv
import csv
from pathlib import Path
import sys

import torch

from torch.utils.data import DataLoader, DistributedSampler
import wandb

from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor
from neuralop.training import setup, AdamW
from neuralop.mpu.comm import get_local_rank
from neuralop.utils import get_wandb_api_key, count_model_params
from neuralop.losses import LpLoss, H1Loss,SoftAdapt

import matplotlib.pyplot as plt
# Read the configuration
from zencfg import make_config_from_cli
import sys
import os
sys.path.insert(0, "../")

from tims.airfrans_all_out.airfrans_trainer import AirfransAllTrainer
from tims.airfrans_all_out.airfrans_dataset_all import load_airfrans_dataset

from tims.airfrans_all_out.config_airfrans_all_weightedLoss import Default
import pandas as pd
import matplotlib.pyplot as plt

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



class WeightedL1Loss(torch.nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0], reduction='sum'):
        super().__init__()
        # Register weights as a buffer so they stay on the correct 4090
        self.register_buffer('weights', torch.tensor(weights).view(1, 4, 1, 1))
        self.reduction = reduction

    def forward(self, pred, y, **kwargs):
        """
        pred: [B, 4, H, W]
        y: [B, 4, H, W]
        """
        # 1. Calculate Absolute Error
        abs_error = torch.abs(pred - y)
        
        # 2. Calculate Target Norm for each channel (normalization factor)
        # We add a small eps to avoid division by zero in the mask regions
        y_norm = torch.abs(y).mean(dim=(-2, -1), keepdim=True) + 1e-6
        
        # 3. Relative Error
        rel_error = abs_error / y_norm
        
        # 4. Apply Channel Weights and take the mean
        # Ensure weights are on the same device as pred (cuda:0 or cuda:1)
        weighted_error = rel_error * self.weights.to(pred.device)
        
        if self.reduction == 'mean':
            return torch.mean(weighted_error)
        else:
            return torch.sum(weighted_error)


class WeightedL2Loss(torch.nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0], reduction='sum'):
        """
        weights: List of weights for [u, v, cp, nut]
        """
        super().__init__()
        # Register as buffer so it moves to GPU with the model automatically
        self.register_buffer('weights', torch.tensor(weights).view(1, 4, 1, 1))
        self.reduction = reduction

    def forward(self, pred, y, **kwargs):
        # Relative Error: |pred - y| / |y|
        # Adding small eps to avoid div by zero
        diff_norm = torch.norm(pred - y, p=2, dim=(-2, -1))
        y_norm = torch.norm(y, p=2, dim=(-2, -1))
        
        rel_error = diff_norm / (y_norm + 1e-6)
        
        # Apply weights to the channel-wise relative error
        weighted_rel_error = rel_error * self.weights.to(pred.device)
        if self.reduction == 'mean':
            return torch.mean(weighted_rel_error)
        else:
            return torch.sum(weighted_rel_error)

# Initialize for your trainer

config = make_config_from_cli(Default)
config = config.to_dict()

# Distributed training setup, if enabled
device, is_logger = setup(config)

# Set up WandB logging
wandb_args = None
if config.wandb.log and is_logger:
    try:
        # Try to login with existing credentials first
        wandb.login()
    except Exception:
        # Fallback to API key file if available
        try:
            wandb.login(key=get_wandb_api_key())
        except Exception as e:
            print(f"Warning: Could not log into WandB: {e}")
            print("Continuing without WandB logging...")
            config.wandb.log = False
    if config.wandb.log and is_logger:
        if config.wandb.name:
            wandb_name = config.wandb.name
        else:
            wandb_name = "_".join(
                f"{var}"
                for var in [
                    config.model.model_arch,
                    config.model.n_layers,
                    config.model.n_modes,
                    config.model.hidden_channels,
                ]
            )
        wandb_args = dict(
            config=config,
            name=wandb_name,
            group=config.wandb.group,
            project=config.wandb.project,
            entity=config.wandb.entity,
        )
        if config.wandb.sweep:
            for key in wandb.config.keys():
                config.params[key] = wandb.config[key]
        wandb.init(**wandb_args)

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print configuration details
if config.verbose and is_logger:
    print(f"##### CONFIG #####\n")
    print(config)
    sys.stdout.flush()

# Load the Airfrans dataset
data_dir = Path(config.data.data_dir).expanduser()

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

# Grab the first batch from your loader
sample = next(iter(train_loader))

# Run through the processor
processed_sample = data_processor.preprocess(sample)
x_norm = processed_sample['x']

# Audit the Mask Channel (index 2)
mask_min = x_norm[:, 2, ...].min().item()
mask_max = x_norm[:, 2, ...].max().item()

print(f"Normalized Mask Range: [{mask_min}, {mask_max}]")
if abs(mask_min) < 1e-5 and abs(mask_max - 1.0) < 1e-5:
    print("✓ VERIFIED: Mask remains strict binary 0/1.")
else:
    print("!! FAILURE: Mask has been scaled. Geometry is corrupted.")

print(f"Logger is set to: {is_logger}")

print ("Output Normalizer Stats:")
if data_processor.out_normalizer is None:
    print("No output normalizer detected.")
else:
    out_m = data_processor.out_normalizer.mean.flatten()
    out_s = data_processor.out_normalizer.std.flatten()
    print(f"Output Stats - Mean: {out_m}, Std: {out_s}")

# Model initialization
model = get_model(config)

# Move model to device
model = model.to(device)
print(f"Model moved to device: {device}")
# Move data processor to device if it has normalizers
if data_processor is not None:
    data_processor = data_processor.to(device)

# convert dataprocessor to an MGPatchingDataProcessor if patching levels > 0
if config.patching.levels > 0:
    data_processor = MGPatchingDataProcessor(
        model=model,
        in_normalizer=data_processor.in_normalizer,
        out_normalizer=data_processor.out_normalizer,
        padding_fraction=config.patching.padding,
        stitching=config.patching.stitching,
        levels=config.patching.levels,
        use_distributed=config.distributed.use_distributed,
        device=device,
    )

# Distributed data parallel setup
# Reconfigure DataLoaders to use a DistributedSampler if in distributed mode
if config.distributed.use_distributed:
    train_db = train_loader.dataset
    train_sampler = DistributedSampler(train_db, rank=get_local_rank())
    train_loader = DataLoader(
        dataset=train_db, batch_size=config.data.batch_size, sampler=train_sampler
    )
    for (res, loader), batch_size in zip(
        test_loaders.items(), config.data.test_batch_sizes
    ):
        test_db = loader.dataset
        test_sampler = DistributedSampler(test_db, rank=get_local_rank())
        test_loaders[res] = DataLoader(
            dataset=test_db, batch_size=batch_size, shuffle=False, sampler=test_sampler
        )
# Create the optimizer
optimizer = AdamW(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

if config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Got scheduler={config.opt.scheduler}")

print(f"Weights for Weighted Loss: {config.data.weights}")
# Create the losses functions
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
weightedL1Loss = WeightedL1Loss(weights=config.data.weights,reduction='sum')  # Custom weights for [u, v, cp]    
weightedL2Loss= WeightedL2Loss(weights=config.data.weights,reduction='sum') 
# Evaluation version (Mean for readable logs)
eval_weighted_l1 = WeightedL1Loss(weights=config.data.weights, reduction='mean')
eval_weighted_l2 = WeightedL2Loss(weights=config.data.weights, reduction='mean')
soft_adapt_loss = SoftAdapt(params=list(model.parameters()), num_losses=4, weights=config.data.weights)
 # Custom weights for [u, v, cp]    
if config.opt.training_loss == "l2":
    train_loss = l2loss
elif config.opt.training_loss == "h1":
    train_loss = h1loss
elif config.opt.training_loss == "weighted_l1":
    train_loss = weightedL1Loss
elif config.opt.training_loss == "weighted_l2":
    train_loss = weightedL2Loss
elif config.opt.training_loss == "soft_adapt":
    train_loss = soft_adapt_loss
else:
    raise ValueError(
        f"Got training_loss={config.opt.training_loss} "
        f'but expected one of ["l2", "h1", "weighted_l1", "weighted_l2", "soft_adapt"]'
    )
eval_losses = {"softadapt": soft_adapt_loss, "h1": h1loss, "l2": l2loss, "weighted_l1": eval_weighted_l1, "weighted_l2": eval_weighted_l2}

if config.verbose and is_logger:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()


# Log model parameter count
if is_logger:
    n_params = count_model_params(model)

    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()

    if config.wandb.log:
        to_log = {"n_params": n_params}
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(to_log, commit=False)
        wandb.watch(model)


## --- Define the outer loop here ---
best_test_loss = float('inf')
improvement_threshold = 0.01 

checkpoint_dir = Path("./tims/airfrans_all_out/checkpoints-all-weighted-L2")
# Ensure the trainer is set to 1 epoch internally
#trainer.n_epochs = 1
output_dir ="/home/timm/Projects/PIML/neuraloperator/tims/airfrans_all_out/results_all_weighted_L2"
        

trainer = AirfransAllTrainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    data_processor=data_processor,
    device=device,
    mixed_precision=config.opt.mixed_precision,
    eval_interval=config.opt.eval_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose,
    wandb_log=config.wandb.log,
)

# Log model parameter count
if is_logger:
    n_params = count_model_params(model)

    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()

    if config.wandb.log:
        to_log = {"n_params": n_params}
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(to_log, commit=False)
        wandb.watch(model)


# Start training process
trainer.train(
    train_loader,
    test_loaders,
    optimizer,
    scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
    save_every=20,
    save_best='128_weighted_l2',  # Save based on weighted L2 at 128 res
    save_dir=checkpoint_dir,
    sample_idx=13,  # Consistent sample for diagnostic plots
)

# Finalize WandB logging
if config.wandb.log and is_logger:
    wandb.finish()

