import torch
import wandb
import sys
from neuralop.training import setup, AdamW
from neuralop import get_model
from neuralop.utils import get_wandb_api_key
from neuralop.losses.data_losses import LpLoss
from neuralop.training.trainer import Trainer
from neuralop.data.datasets import CarOTDataset, load_saved_ot, CFDDataProcessor
from copy import deepcopy

# query points is [sdf_query_resolution] * 3 (taken from config ahmed)
# Read the configuration
from zencfg import make_config_from_cli
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from config.otno_carcfd_config import Default

config = make_config_from_cli(Default)
config = config.to_dict()

#Set-up distributed communication, if using
device, is_logger = setup(config)

#Set up WandB logging
wandb_init_args = {}
config_name = "car-pressure-otno"
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}" for var in [config_name, config.data.expand_factor]
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

#Load CFD body data (once the OT dataset is generated, we can use "load_saved_ot", avoiding recomputation)
data_module = CarOTDataset(
    root_dir=config.data.root,
    n_train=config.data.n_train,
    n_test=config.data.n_test,
    expand_factor=config.data.expand_factor,
    reg=config.data.reg,
    device=device,
)

train_loader = data_module.train_loader(batch_size=1, shuffle=True)
test_loader = data_module.test_loader(batch_size=1, shuffle=False)

model = get_model(config)

#Create the optimizer
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
    raise ValueError(f"Got {config.opt.scheduler=}")


l2loss = LpLoss(d=2, p=2)

if config.opt.training_loss == "l2":
    train_loss_fn = l2loss
else:
    raise ValueError(f"Got {config.opt.training_loss=}")

if config.opt.testing_loss == "l2":
    test_loss_fn = l2loss
else:
    raise ValueError(f"Got {config.opt.testing_loss=}")

output_encoder = deepcopy(data_module.normalizers["press"]).to(device)
data_processor = CFDDataProcessor(normalizer=output_encoder, device=device)

trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    data_processor=data_processor,
    device=device,
    wandb_log=config.wandb.log,
    verbose=is_logger,
)

if config.wandb.log:
    wandb.log({"time_to_distance": data_module.time_to_distance}, commit=False)

trainer.train(
    train_loader=train_loader,
    test_loaders={"": test_loader},
    optimizer=optimizer,
    scheduler=scheduler,
    training_loss=train_loss_fn,
    eval_losses={config.opt.testing_loss: test_loss_fn},
    regularizer=None,
)
