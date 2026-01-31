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

from tims.airfrans.airfrans_dataset import load_airfrans_dataset
import matplotlib.pyplot as plt
# Read the configuration
from zencfg import make_config_from_cli
import sys
import os
sys.path.insert(0, "../")
from tims.airfrans.airfrans_config_cp_weightedLoss import Default
import pandas as pd
import matplotlib.pyplot as plt

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def plot_convergence(csv_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['u_err'], label='U-Velocity Error %')
    plt.plot(df['epoch'], df['v_err'], label='V-Velocity Error %')
    plt.plot(df['epoch'], df['cp_err'], label='Cp Error %')
    plt.yscale('log') # Log scale is best for observing convergence plateaus
    plt.xlabel('Epoch')
    plt.ylabel('Relative Error (%)')
    plt.title('Channel-Wise Convergence Audit')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

def validation_plot_hook(model, dataset, data_processor, device, epoch, idx, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "channel_convergence_log.csv")
    model.eval()
    
    # 1. Grab a consistent sample
    batch = dataset[idx]
    x_raw = batch['x'].unsqueeze(0).to(device)  
    y_raw = batch['y'].unsqueeze(0).to(device)  
    
    with torch.no_grad():
        # 2. Preprocess & Input Signal Audit
        processed = data_processor.preprocess({'x': x_raw, 'y': y_raw})
        x_norm = processed['x']
        y_norm_truth = processed['y']
        
        # 3. Model Inference
        y_norm_pred = model(x_norm)
        
        # 4. Error Metrics & Logging
        out_labels_err = ['u_err', 'v_err', 'cp_err']
        row_data = {'epoch': epoch, 
                    'sdf_min': x_norm[0, 3].min().item(), 
                    'sdf_max': x_norm[0, 3].max().item()}
        
        for i, label in enumerate(out_labels_err):
            abs_diff = torch.abs(y_norm_pred[0, i] - y_norm_truth[0, i]).mean()
            target_mag = torch.abs(y_norm_truth[0, i]).mean() + 1e-6
            rel_err = (abs_diff / target_mag).item() * 100
            row_data[label] = rel_err
        
        # 5. CSV Logging
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'u_err', 'v_err', 'cp_err', 'sdf_min', 'sdf_max'])
            if not file_exists: writer.writeheader()
            writer.writerow(row_data)        

        # 6. Back to Physics
        y_phys_pred = data_processor.postprocess(y_norm_pred)

    # --- PLOTTING LOGIC ---
    out_labels = ['U-Velocity', 'V-Velocity', 'Pressure (Cp)']
    in_labels = ['Mask (Ch 2)', 'SDF (Ch 3)']
    
    # Grid: 3 Rows (Outputs) x 7 Columns (2 Input Audit + 5 Output Audit)
    fig, axes = plt.subplots(3, 7, figsize=(35, 12))
    plt.suptitle(f"Epoch {epoch}: Input & Output Signal Audit", fontsize=20)

    for i in range(3):
        # --- INPUT AUDIT (Columns 0-1) ---
        if i < 2: # Only 2 interesting geometry inputs: Mask and SDF
            # Column 0: Raw Input
            im_in_r = axes[i, 0].imshow(x_raw[0, i+2].cpu(), origin='lower')
            plt.colorbar(im_in_r, ax=axes[i, 0])
            axes[i, 0].set_title(f"Raw {in_labels[i]}")

            # Column 1: Normalized Input (What the model actually sees)
            im_in_n = axes[i, 1].imshow(x_norm[0, i+2].cpu(), origin='lower', cmap='RdBu_r')
            plt.colorbar(im_in_n, ax=axes[i, 1])
            axes[i, 1].set_title(f"Norm {in_labels[i]}")
        else:
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')

        # --- OUTPUT AUDIT (Columns 2-6) ---
        # Column 2: Truth Phys
        im2 = axes[i, 2].imshow(y_raw[0, i].cpu(), origin='lower')
        plt.colorbar(im2, ax=axes[i, 2])
        axes[i, 2].set_title(f"Truth Phys: {out_labels[i]}")

        # Column 3: Truth Norm (Z)
        im3 = axes[i, 3].imshow(y_norm_truth[0, i].cpu(), origin='lower', cmap='plasma')
        plt.colorbar(im3, ax=axes[i, 3])
        axes[i, 3].set_title(f"Truth Norm (Z)")

        # Column 4: Pred Norm (Z)
        vz_min, vz_max = y_norm_truth[0, i].min().item(), y_norm_truth[0, i].max().item()
        im4 = axes[i, 4].imshow(y_norm_pred[0, i].cpu(), origin='lower', cmap='plasma', vmin=vz_min, vmax=vz_max)
        plt.colorbar(im4, ax=axes[i, 4])
        axes[i, 4].set_title(f"Pred Norm (Z)")

        # Column 5: Phys Pred
        vp_min, vp_max = y_raw[0, i].min().item(), y_raw[0, i].max().item()
        im5 = axes[i, 5].imshow(y_phys_pred[0, i].cpu(), origin='lower', vmin=vp_min, vmax=vp_max)
        plt.colorbar(im5, ax=axes[i, 5])
        axes[i, 5].set_title(f"Phys Pred")

        # Column 6: Phys Residual
        res = y_raw[0, i] - y_phys_pred[0, i]
        im6 = axes[i, 6].imshow(res.cpu(), origin='lower', cmap='RdBu_r')
        plt.colorbar(im6, ax=axes[i, 6])
        axes[i, 6].set_title(f"Residual")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f"audit_epoch_{epoch:04d}.png")
    plt.savefig(save_path)
    plt.close(fig)

def verify_input_encoder(encoder):
    print(f"\n{'='*20} INPUT ENCODER AUDIT {'='*20}")
    
    # 1. Check Channel Dimensions
    mean = encoder.mean.flatten()
    std = encoder.std.flatten()
    print(f"Stats Shape: {list(encoder.mean.shape)} | Channels detected: {len(mean)}")

    # 2. Check Physical Mapping
    # We expect 3 channels of stats representing [u_inf, v_inf, sdf]
    # which will be applied to indices [0, 1, 3] of the 4D input.
    names = ["u_velocity (inf)", "v_velocity (inf)", "SDF (geometry)"]
    
    print(f"\n{'Channel':<20} | {'Mean':>10} | {'Std':>10}")
    print("-" * 45)
    for i, name in enumerate(names):
        m, s = mean[i].item(), std[i].item()
        print(f"{name:<20} | {m:>10.4f} | {s:>10.4f}")

    # 3. Verify Selective Logic
    channels = getattr(encoder, 'channels_to_normalize', [])
    print(f"\nActive Channels for Normalization: {channels}")
    
    if 2 in channels:
        print("!! WARNING: Channel 2 (Mask) is set to be normalized! This will corrupt geometry.")
    else:
        print("✓ SUCCESS: Channel 2 (Mask) will be passed through untouched.")
    print(f"{'='*63}\n")

def save_checkpoint(checkpoint_dir, model, data_processor, epoch, filename):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Handle DDP: get the internal module
    unwrapped_model = model.module if hasattr(model, 'module') else model
    
    torch.save({
        'epoch': epoch,
        'model': unwrapped_model.state_dict(),
        'data_processor': data_processor.state_dict(),
        'config': config
    }, checkpoint_dir / filename)

class WeightedL1Loss(torch.nn.Module):
    def __init__(self, weights=[3.0, 3.0, 1.0], reduction='sum'):
        super().__init__()
        # Register weights as a buffer so they stay on the correct 4090
        self.register_buffer('weights', torch.tensor(weights).view(1, 3, 1, 1))
        self.reduction = reduction

    def forward(self, pred, y, **kwargs):
        """
        pred: [B, 3, H, W]
        y: [B, 3, H, W]
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
    def __init__(self, weights=[2.0, 2.0, 1.0], reduction='sum'):
        """
        weights: List of weights for [u, v, Cp]
        """
        super().__init__()
        # Register as buffer so it moves to GPU with the model automatically
        self.register_buffer('weights', torch.tensor(weights).view(1, 3, 1, 1))
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
# Giving Cp (index 2) a weight of 2.0 or 3.0 is a common starting point

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
        encode_input=config.data.encode_input,    
        encode_output=config.data.encode_output, 
        encoding=config.data.encoding,
        channel_dim=1,
    )

# check data_processor

verify_input_encoder(data_processor.in_normalizer)


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
weightedL1Loss = WeightedL1Loss(weights=config.data.weights,reduction='sum')  # Custom weights for [u, v, Cp]    
weightedL2Loss= WeightedL2Loss(weights=config.data.weights,reduction='sum') 
# Evaluation version (Mean for readable logs)
eval_weighted_l1 = WeightedL1Loss(weights=config.data.weights, reduction='mean')
eval_weighted_l2 = WeightedL2Loss(weights=config.data.weights, reduction='mean')
 # Custom weights for [u, v, Cp]    
if config.opt.training_loss == "l2":
    train_loss = l2loss
elif config.opt.training_loss == "h1":
    train_loss = h1loss
elif config.opt.training_loss == "weighted_l1":
    train_loss = weightedL1Loss
elif config.opt.training_loss == "weighted_l2":
    train_loss = weightedL2Loss
else:
    raise ValueError(
        f"Got training_loss={config.opt.training_loss} "
        f'but expected one of ["l2", "h1", "weighted_l1", "weighted_l2"]'
    )
eval_losses = {"h1": h1loss, "l2": l2loss, "weighted_l1": eval_weighted_l1, "weighted_l2": eval_weighted_l2}

if config.verbose and is_logger:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()

trainer = Trainer(
    model=model,
    n_epochs=1,
    device=device,
    data_processor=data_processor,
    mixed_precision=config.opt.mixed_precision,
    wandb_log=config.wandb.log,
    eval_interval=config['opt'].get('eval_interval', 5),
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose and is_logger,
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


## --- Define the outer loop here ---
best_test_loss = float('inf')
improvement_threshold = 0.01 

checkpoint_dir = Path("./tims/airfrans/checkpoints-weighted-L2")
# Ensure the trainer is set to 1 epoch internally
trainer.n_epochs = 1
output_dir ="/home/timm/Projects/PIML/neuraloperator/tims/airfrans/results_Cp_weighted_L2"

#device = torch.device('cuda:1')
print(f" Logger {is_logger} Training for {config['opt']['n_epochs']} epochs...")
for epoch in range(config['opt']['n_epochs']):
    # Capture the metrics returned by the trainer
    # This captures: (avg_loss, avg_error, eval_metrics_dict)
    # Use *metadata to catch any 4th, 5th, or extra return values
# 1. Capture the entire output as a dictionary
    metrics = trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses
    )

    print(f"--- Completed Epoch {epoch+1}/{config['opt']['n_epochs']} ---")
    print(f"Metrics: {metrics}")
    # Now 'epoch' is available here for your custom logic
 
    if is_logger and config.wandb.log:
        # 1. Get the current loss from the trainer's internal state
        # Usually stored in trainer.test_losses
        res_key = f"{config['data']['test_resolutions'][0]}_l2"

        # Pull the loss from the eval_metrics returned by the trainer
        current_test_loss = metrics.get(res_key, float('inf'))
        wandb.log(metrics,step=epoch)
        # Improvement check
        improvement = (best_test_loss - current_test_loss) / best_test_loss if best_test_loss != float('inf') else 1.0
        
        if improvement >= improvement_threshold:
            best_test_loss = current_test_loss
            save_checkpoint(checkpoint_dir, model, data_processor, epoch, "airfrans_cp_l2_fno_best.pt")
            print(f">>> Epoch {epoch}: NEW BEST | {res_key}: {best_test_loss:.4f} | Imp: {improvement:.2%}")
            
        if epoch % config['opt'].get('save_interval', 50) == 0:
            test_loader_key = list(test_loaders.keys())[0]
            dataset = test_loaders[test_loader_key].dataset
            validation_plot_hook(model, dataset, data_processor, device, epoch, 13, output_dir=output_dir)
            save_checkpoint(checkpoint_dir, model, data_processor, epoch, f"checkpoint_epoch_{epoch}.pt")

    model_final = "airfrans_cp_l2_fno_final.pt"
    save_checkpoint(checkpoint_dir, model, data_processor, epoch, model_final)



# Finalize WandB logging
if config.wandb.log and is_logger:
    # Log the checkpoint files as artifacts
    artifact = wandb.Artifact("model-checkpoint", type="model")
    artifact.add_file(str(checkpoint_dir / f"airfrans_cp_l2_fno_final.pt"))
    #artifact.add_file(str(checkpoint_dir / f"airfrans_cp_l2_fno_model.pt"))
    wandb.log_artifact(artifact)
    
    wandb.finish()

log_file = os.path.join(output_dir, "channel_convergence_log.csv")

plot_convergence(log_file)