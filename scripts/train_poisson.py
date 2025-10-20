"""
Training script for nonlinear Poisson equation using GINO.

This script trains a Graph Neural Operator (GINO) on the nonlinear Poisson
equation with boundary conditions. The model learns to solve elliptic PDEs
with complex geometries and boundary conditions using graph-based representations.
"""

import sys

import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

# Set warning filter to show each warning only once
import warnings
warnings.filterwarnings("once", category=UserWarning)

from neuralop.losses.data_losses import LpLoss, MSELoss
from neuralop.training import Trainer, setup
from neuralop.data.datasets.nonlinear_poisson import load_nonlinear_poisson_pt
from neuralop.losses.equation_losses import PoissonBoundaryLoss, PoissonEqnLoss
from neuralop.losses.meta_losses import WeightedSumLoss
from neuralop.models import get_model
from neuralop.utils import get_wandb_api_key, count_model_params

# Configuration setup
config_name = "default"
from zencfg import make_config_from_cli
import sys

sys.path.insert(0, "../")
from config.poisson_gino_config import Default

config = make_config_from_cli(Default)
config = config.to_dict()

# Distributed training setup, if enabled
device, is_logger = setup(config)

# WandB logging configuration
wandb_args = None
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.model.in_gno_radius,
                config.model.out_gno_radius,
                config.model.gno_weighting_function,
                config.model.gno_weight_function_scale,
                config.model.fno_n_modes,
                config.model.fno_n_layers,
                config.data.n_train,
                config.data.n_test,
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

else:
    wandb_init_args = None
# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print configuration details
if config.verbose:
    print(f"##### CONFIG #####\n")
    print(config)
    sys.stdout.flush()

# Load the nonlinear Poisson dataset
train_loader, test_loader, data_processor = load_nonlinear_poisson_pt(
    data_path=config.data.file,
    query_res=config.data.query_resolution,
    n_train=config.data.n_train,
    n_test=config.data.n_test,
    n_in=config.data.n_in,
    n_out=config.data.n_out,
    n_eval=config.data.n_eval,
    n_bound=config.data.n_bound,
    val_on_same_instance=config.data.single_instance,
    train_out_res=config.data.train_out_res,
    input_min_sample_points=config.data.input_min,
    input_max_sample_points=config.data.input_max,
    input_subsample_level=config.data.sample_random_in,
    output_subsample_level=config.data.sample_random_out,
    return_dict=config.data.return_queries_dict,
)

# Create test loaders dictionary
test_loaders = {"test": test_loader}

# Model initialization
model = get_model(config)
model = model.to(device)

# Distributed data parallel setup
if config.distributed.use_distributed:
    model = DDP(
        model, device_ids=[device.index], output_device=device.index, static_graph=True
    )

# Create the optimizer
optimizer = torch.optim.Adam(
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

# Create the losses: MSE and relative L2
mse_loss = MSELoss()
l2_loss = LpLoss(d=2, p=2)


class GINOLoss(object):
    """
    Custom loss wrapper for GINO models that handle dictionary outputs.

    This loss function concatenates dictionary outputs from GINO models
    before applying the base loss function.
    """

    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss

    def __call__(self, out, y, **kwargs):
        """
        Apply loss to GINO model outputs.

        Parameters
        ----------
        out : dict or torch.Tensor
            Model output, either dictionary of field outputs or tensor
        y : dict or torch.Tensor
            Ground truth, either dictionary of field targets or tensor
        **kwargs
            Additional arguments passed to base loss

        Returns
        -------
        torch.Tensor
            Computed loss value
        """
        if kwargs:
            warnings.warn(
                f"GINOLoss.__call__() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2
            )
        if isinstance(out, dict) and isinstance(y, dict):
            y = torch.cat([y[field] for field in out.keys()], dim=1)
            out = torch.cat([out[field] for field in out.keys()], dim=1)

        return self.base_loss(out, y, **kwargs)


gino_mseloss = GINOLoss(mse_loss)

training_loss = config.opt.training_loss
if not isinstance(training_loss, (tuple, list)):
    training_loss = [training_loss]

losses = []
weights = []

if "mse" in training_loss:
    losses.append(gino_mseloss)
    weights.append(config.opt.loss_weights.get("mse", 1.0))
if "equation" in training_loss:
    equation_loss = PoissonEqnLoss(
        interior_weight=config.opt.loss_weights.get("interior", 1.0),
        boundary_weight=config.opt.loss_weights.get("boundary", 1.0),
        diff_method=config.opt.get("pino_method", "autograd"),
    )
    losses.append(equation_loss)
    weights.append(1)

if len(losses) == 1:
    train_loss = losses[0]
else:
    train_loss = WeightedSumLoss(losses=losses, weights=weights)

eval_losses = {"mse": mse_loss, "relative_l2": l2_loss}

if config.verbose:
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
        wandb.log(to_log)
        wandb.watch(model)

# Start training process
trainer.train(
    train_loader,
    test_loaders,
    optimizer,
    scheduler,
    regularizer=None,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

# Finalize WandB logging
if config.wandb.log and is_logger:
    wandb.finish()
