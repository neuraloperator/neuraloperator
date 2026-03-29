"""
Training script for Burgers equation using the Recurrent Neural Operator (RNO).

This script trains an RNO on the 1D time-dependent Burgers equation
using a small in-repo dataset.
"""

import sys
from pathlib import Path

import torch
import wandb
import torch.nn as nn

from zencfg import make_config_from_cli

# Add project root to sys.path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.burgers_rno_config import Default
from neuralop.utils import get_wandb_api_key, count_model_params
from neuralop.training import AdamW, Trainer
from neuralop.losses import LpLoss
from neuralop.models import get_model
from neuralop.data.datasets import load_mini_burgers_1dtime


class RNOTimeAdapter(nn.Module):
    """Adapter to reconcile dataset batch format with RNO's expected input format.

    The mini_burgers_1dtime dataset returns batches with shape (B, C, T, X),
    where the time dimension is at index 2. However, RNO expects the time
    dimension at index 1 with shape (B, T, C, X). This adapter performs the
    necessary permutation and extracts the final time step prediction.

    This wrapper is necessary because the Trainer expects a standard forward(x, y)
    signature that returns predictions matching the target format, whereas the raw
    RNO model returns both predictions and hidden states with a different input format.

    Parameters
    ----------
    core : nn.Module
        The RNO model to wrap. This is the trained RNO that performs the
        actual sequence modeling.

    Input
    -----
    x : torch.Tensor
        Input with shape (B, C, T, X) from the dataset.

    Returns
    -------
    torch.Tensor
        Prediction of the last time step with shape (B, C, X).
    """

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, x, y=None):
        # (B, C, T, X) -> (B, T, C, X)
        x_rno = x.permute(0, 2, 1, 3).contiguous()
        pred = self.core(x_rno)
        return pred


class LastFrameDataProcessor(nn.Module):
    """Processor that extracts the last time frame from targets for single-step supervision.

    The Burgers dataset provides full temporal sequences as targets with shape
    (B, C, T, X), but we train the RNO to predict only the final time step.
    This processor wraps an existing data_processor and extracts y[:, :, -1, :]
    during preprocessing, converting the target to shape (B, C, X) for L2 loss computation.

    Parameters
    ----------
    base : nn.Module or object
        The underlying data processor to wrap (e.g., DefaultDataProcessor).
        This processor's preprocess/postprocess methods are called before/after
        extracting the last frame.
    """

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.device = "cpu"

    def to(self, device):
        self.device = device
        if hasattr(self.base, "to"):
            self.base = self.base.to(device)
        return self

    def preprocess(self, data_dict, batched=True):
        if hasattr(self.base, "preprocess"):
            data_dict = self.base.preprocess(data_dict, batched=batched)
        # Supervise on last time step only
        if torch.is_tensor(data_dict["y"]):
            data_dict["y"] = data_dict["y"][:, :, -1, :]
        return data_dict

    def postprocess(self, output, data_dict):
        if hasattr(self.base, "postprocess"):
            return self.base.postprocess(output, data_dict)
        return output, data_dict


def main():
    config = make_config_from_cli(Default)
    config = config.to_dict()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up WandB logging
    if config.wandb.log:
        wandb.login(key=get_wandb_api_key())
        if config.wandb.name:
            wandb_name = config.wandb.name
        else:
            wandb_name = "_".join(
                f"{var}" for var in [
                    config["arch"],
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

    # Print configuration details
    if config.verbose:
        print("##### CONFIG ######")
        print(config)
        sys.stdout.flush()

    # Data loading
    data_path = PROJECT_ROOT / config.data.folder
    train_loader, test_loaders, data_processor = load_mini_burgers_1dtime(
        data_path=data_path,
        n_train=config.data.n_train,
        batch_size=config.data.batch_size,
        n_test=config.data.n_tests[0],
        test_batch_size=config.data.test_batch_sizes[0],
        temporal_subsample=config.data.get("temporal_subsample", 1),
        spatial_subsample=config.data.get("spatial_subsample", 1),
    )

    # Model instantiation and wrapping
    core_model = get_model(config)
    model = RNOTimeAdapter(core_model).to(device)

    # Data processor wrapper: convert target to last time step
    if data_processor is not None:
        data_processor = LastFrameDataProcessor(data_processor).to(device)

    # Optimizer and scheduler
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

    # Losses
    l2_loss = LpLoss(d=2, p=2)
    eval_losses = {"l2": l2_loss}
    train_loss = l2_loss

    if config.verbose:
        print("\n### MODEL ###\n", model)
        print("\n### OPTIMIZER ###\n", optimizer)
        print("\n### SCHEDULER ###\n", scheduler)
        print("\n### LOSSES ###")
        print(f"\n * Train: {train_loss}")
        print(f"\n * Test: {eval_losses}")
        print(f"\n### Beginning Training...\n")
        sys.stdout.flush()

    # Trainer
    trainer = Trainer(
        model=model,
        n_epochs=config.opt.n_epochs,
        device=device,
        data_processor=data_processor,
        mixed_precision=config.opt.mixed_precision,
        wandb_log=config.wandb.log,
        eval_interval=config.opt.eval_interval,
        log_output=config.wandb.log_output,
        use_distributed=config.distributed.use_distributed,
        verbose=config.verbose,
    )

    # Log model parameter count
    n_params = count_model_params(model)
    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()
    if config.wandb.log:
        wandb.log({"n_params": n_params}, commit=False)
        wandb.watch(model)

    # Start training
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
    )

    if config.wandb.log:
        wandb.finish()


if __name__ == "__main__":
    main()
