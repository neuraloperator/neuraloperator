import sys
import torch
import wandb
import torch.nn.functional as F

from neuralop import H1Loss, LpLoss, BurgersEqnLoss, ICLoss, get_model
from neuralop.data.datasets import load_mini_burgers_1dtime
from neuralop.training import AdamW
from neuralop.utils import get_wandb_api_key, count_model_params, get_project_root
from neuralop.losses.meta_losses import Relobralo, SoftAdapt

# Read the configuration
config_name = "default"
from zencfg import make_config_from_cli 
import sys 
sys.path.insert(0, '../')
from config.burgers_pino_config import Default

config = make_config_from_cli(Default)
config = config.to_dict()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Set up WandB logging
if config.wandb.log:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.model.model_arch,
                config.model.n_layers,
                config.model.n_modes,
                config.model.hidden_channels,
            ]
        )
    wandb_init_args = dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_init_args)
else: 
    wandb_init_args = None


# Print config
if config.verbose:
    print("##### CONFIG ######")
    print(config)
    sys.stdout.flush()

data_path = get_project_root() / config.data.folder
# Load the Burgers dataset
train_loader, test_loaders, data_processor = load_mini_burgers_1dtime(data_path=data_path,
        n_train=config.data.n_train, batch_size=config.data.batch_size, 
        n_test=config.data.n_tests[0], test_batch_size=config.data.test_batch_sizes[0],
        temporal_subsample=config.data.get("temporal_subsample", 1),
        spatial_subsample=config.data.get("spatial_subsample", 1),
        )

# Create the model
model = get_model(config)
model = model.to(device)

# Create the optimizer
optimizer = AdamW(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

# Create the scheduler
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

# Create the losses
l2_loss = LpLoss(d=2, p=2)
h1_loss = H1Loss(d=2)
ic_loss = ICLoss()
equation_loss = BurgersEqnLoss(method=config.opt.get('pino_method', 'fdm'), visc=0.01, loss=F.mse_loss)

loss_map = {
    'l2': l2_loss,
    'h1': h1_loss,
    'ic': ic_loss,
    'equation': equation_loss
}

training_losses = [loss_map[name] for name in config.opt.training_loss]

# Create loss aggregator
if config.opt.loss_aggregator.lower() == 'relobralo':
    train_loss = Relobralo(
        num_losses=len(training_losses),
        params=model.parameters(),
        alpha=0.5,
        beta=0.9,
        tau=1.0
    )
elif config.opt.loss_aggregator.lower() == 'softadapt':
    train_loss = SoftAdapt(
        num_losses=len(training_losses),
        params=model.parameters(),
    )
else:
    raise ValueError(f"Unknown loss_aggregator: {config.opt.loss_aggregator}. Use 'relobralo' or 'softadapt'.")

# Evaluation losses
eval_losses = {
    "h1": h1_loss,
    "l2": l2_loss
}

if config.verbose:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()

# Log number of parameter
if config.verbose:
    n_params = count_model_params(model)
    print(f"\nn_params: {n_params}")
    sys.stdout.flush()
if config.wandb.log:
    wandb.log({"n_params": n_params}, commit=False)
    wandb.watch(model)



# Training loop
model.train()
for epoch in range(config.opt.n_epochs):
    model.train()
    train_losses = []
    loss_values = {name: [] for name in config.opt.training_loss}
    
    # Training batches
    for batch_idx, sample in enumerate(train_loader):
        
        # Move tensors to device
        sample = {k: v.to(device).float() if torch.is_tensor(v) else v for k, v in sample.items()}
        
        optimizer.zero_grad(set_to_none=True)
        
        # Preprocess data
        if data_processor is not None:
            sample = data_processor.preprocess(sample)
            sample = {k: v.to(device).float() if torch.is_tensor(v) else v for k, v in sample.items()}
        
        # Forward pass
        pred = model(**sample)
        
        # Postprocess output
        if data_processor is not None:
            pred, sample = data_processor.postprocess(pred, sample)
            sample = {k: v.to(device).float() if torch.is_tensor(v) else v for k, v in sample.items()}
        
        # Compute individual losses
        loss_vals = {}
        for loss_name in config.opt.training_loss:
            if loss_name == 'equation':
                loss_val = loss_map[loss_name](pred, x=sample['x'])
            else:
                loss_val = loss_map[loss_name](pred, sample['y'])
            
            loss_vals[loss_name] = loss_val 
            loss_values[loss_name].append(loss_val.item())
        
        # Compute total loss using loss aggregator
        total_loss, weights = train_loss(loss_vals, step=epoch)

        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Record losses
        train_losses.append(total_loss.item())
    
    # Calculate average losses
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_losses = {name: sum(vals) / len(vals) for name, vals in loss_values.items()}
    
    # Print training progress
    if config.verbose:
        print(f"\nEpoch {epoch+1}/{config.opt.n_epochs}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print("Individual Training Losses:")
        for name, avg_val in avg_losses.items():
            print(f"  {name.upper()} Loss: {avg_val:.4f}")
        
        if hasattr(train_loss, 'weights'):
            print("Loss Weights:")
            for i, name in enumerate(config.opt.training_loss):
                print(f"  {name.upper()} Weight: {weights[i]:.4f}")
        sys.stdout.flush()
    
    # Evaluation
    eval_losses_dict = {}
    if epoch % config.opt.eval_interval == 0:
        model.eval()
        
        with torch.no_grad():
            for test_name, test_loader in test_loaders.items():
                test_losses = []
                for sample in test_loader:
                    sample = {k: v.to(device).float() if torch.is_tensor(v) else v for k, v in sample.items()}
                    
                    if data_processor is not None:
                        sample = data_processor.preprocess(sample)
                        sample = {k: v.to(device).float() if torch.is_tensor(v) else v for k, v in sample.items()}
                    
                    pred = model(**sample)
                    
                    if data_processor is not None:
                        pred, sample = data_processor.postprocess(pred, sample)
                        sample = {k: v.to(device).float() if torch.is_tensor(v) else v for k, v in sample.items()}
                    
                    l2_test_loss = l2_loss(pred.float(), sample['y'].float())
                    h1_test_loss = h1_loss(pred.float(), sample['y'].float())
                    test_losses.append(l2_test_loss.item())
                
                avg_test_loss = sum(test_losses) / len(test_losses)
                eval_losses_dict[f"{test_name}_loss"] = avg_test_loss
                
                if config.verbose:
                    print(f"\nTest {test_name} loss: {avg_test_loss:.4f}")
    
    # Update learning rate
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(eval_losses_dict.get("test_loss", avg_train_loss))
    else:
        scheduler.step()
    
    # Log everything
    if config.wandb.log:
        log_dict = {
            "train_loss": avg_train_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        
        # Log individual losses
        for name, avg_val in avg_losses.items():
            log_dict[f"train_{name}_loss"] = avg_val
        
        if hasattr(train_loss, 'weights'):
            for i, name in enumerate(config.opt.training_loss):
                log_dict[f"weight_{name}"] = weights[i]
            
        if eval_losses_dict:
            log_dict.update(eval_losses_dict)
            
        wandb.log(log_dict)
    
    if device.type == 'cuda' and epoch % 10 == 0:
        torch.cuda.empty_cache()

if config.wandb.log:
    wandb.finish() 