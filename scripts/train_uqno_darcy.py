import sys

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.datasets import load_darcy_flow_small
from neuralop.datasets.data_transforms import DataProcessor, MGPatchingDataProcessor
from neuralop.losses import PointwiseQuantileLoss
from neuralop.models import UQNO
from neuralop.training import setup
from neuralop.training.callbacks import BasicLoggerCallback, Callback
from neuralop.utils import get_wandb_api_key, count_model_params


# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./uqno_config.yaml", config_name="default", config_folder="../config"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="../config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

# Set-up distributed communication, if using
device, is_logger = setup(config)

# Set up WandB logging
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
                config.tfno2d.n_layers,
                config.tfno2d.hidden_channels,
                config.tfno2d.n_modes_width,
                config.tfno2d.n_modes_height,
                config.tfno2d.factorization,
                config.tfno2d.rank,
                config.patching.levels,
                config.patching.padding,
            ]
        )
    wandb_args =  dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print config to screen
if config.verbose and is_logger:
    pipe.log()
    sys.stdout.flush()

# Loading the Darcy flow dataset for training the base model
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=config.data.n_train_solution,
    batch_size=config.data.batch_size,
    positional_encoding=config.data.positional_encoding,
    test_resolutions=config.data.test_resolutions,
    n_tests=config.data.n_tests,
    test_batch_sizes=config.data.test_batch_sizes,
    encode_input=False,
    encode_output=False,
)
# convert dataprocessor to an MGPatchingDataprocessor if patching levels > 0
if config.patching.levels > 0:
    data_processor = MGPatchingDataProcessor(in_normalizer=data_processor.in_normalizer,
                                             out_normalizer=data_processor.out_normalizer,
                                             positional_encoding=data_processor.positional_encoding,
                                             padding_fraction=config.patching.padding,
                                             stitching=config.patching.stitching,
                                             levels=config.patching.levels)
data_processor = data_processor.to(device)

solution_model = get_model(config)
solution_model = solution_model.to(device)

# Use distributed data parallel
if config.distributed.use_distributed:
    model = DDP(
        solution_model, device_ids=[device.index], output_device=device.index, static_graph=True
    )

# Create the optimizer
optimizer = torch.optim.Adam(
    solution_model.parameters(),
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


# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
if config.opt.training_loss == "l2":
    train_loss = l2loss
elif config.opt.training_loss == "h1":
    train_loss = h1loss
else:
    raise ValueError(
        f'Got training_loss={config.opt.training_loss} '
        f'but expected one of ["l2", "h1"]'
    )
eval_losses = {"h1": h1loss, "l2": l2loss}

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
    model=solution_model,
    n_epochs=config.opt.n_epochs,
    device=device,
    data_processor=data_processor,
    amp_autocast=config.opt.amp_autocast,
    wandb_log=config.wandb.log,
    log_test_interval=config.wandb.log_test_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose and is_logger,
    callbacks=[
        BasicLoggerCallback(wandb_args)
              ]
              )

# Log parameter count
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

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

######
# UQ #
######

## TODO
# compute quantile loss as follows:
# y = solution(x) - y_true
# x = residual(x)

# quantile(x,y) is pointwise quantile loss

# compute via data processor


class QuantileLossDataProcessor(DataProcessor):
    def __init__(self, base_data_processor: DataProcessor,
                 device: str="cpu"):
        """QuantileLossDataProcessor converts tuple (G_hat(a,x), E(a,x)) and 
        sample['y'] = G_true(a,x) into the form expected by PointwiseQuantileLoss

        y_pred = E(a,x)
        y_true = abs(G_hat(a,x) - G_true(a,x))

        It also preserves any transformations that need to be performed
        on inputs/outputs from the solution model. 

        Parameters
        ----------
        base_data_processor : DataProcessor
            transforms required for base solution_model input/output
        device: str
            "cpu" or "cuda" 
        """
        super().__init__()
        self.base_data_processor = base_data_processor
        self.device = device
    
    def wrap(self, model):
        self.model = model
        return self

    def to(self, device):
        self.device = device
        self.base_data_processor = self.base_data_processor.to(device)
        return self

    def preprocess(self, *args, **kwargs):
        """
        nothing required at preprocessing - just wrap the base DataProcessor
        """
        return self.base_data_processor.preprocess(*args, **kwargs)
    
    def postprocess(self, out, sample):
        """
        wrap the base_data_processor's outputs and transform
        """
        out, sample = self.base_data_processor.postprocess(out, sample)
        g_true = sample['y']
        g_hat, pred_uncertainty_ball = out # UQNO returns a tuple
        sample['y'] = g_true - g_hat
        sample.pop('x') # remove x arg to avoid overloading loss args
        return pred_uncertainty_ball, sample

    def forward(self, **sample):
        # combine pre and postprocess for wrap
        sample = self.preprocess(sample)
        out = self.model(**sample)
        out, sample = self.postprocess(out, sample)
        return out, sample

uqno = UQNO(base_model=solution_model)
quantile_loss = PointwiseQuantileLoss(alpha=config.opt.alpha)
quantile_data_proc = QuantileLossDataProcessor(base_data_processor=data_processor,
                                               device=device)

# Create the quantile model's optimizer
quantile_optimizer = torch.optim.Adam(
    uqno.residual_model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

# reuse scheduler

quantile_trainer = Trainer(model=uqno,
                           n_epochs=config.opt.n_epochs,
                           data_processor=quantile_data_proc,
                           wandb_log=config.wandb.log,
                           device=device,
                           amp_autocast=config.opt.amp_autocast,
                           log_test_interval=config.wandb.log_test_interval,
                           log_output=config.wandb.log_output,
                           use_distributed=config.distributed.use_distributed,
                           verbose=config.verbose and is_logger,
                           callbacks=[
                                BasicLoggerCallback(wandb_args)
                                    ]
                           )

quantile_trainer.train(train_loader=quantile_train_loader, # add this later
                       test_loaders={}, # no eval on quantile train
                       optimizer=quantile_optimizer,
                       scheduler=scheduler,
                       training_loss=quantile_loss,
                       )

### TODO:  calibrate trained quantile model

if config.wandb.log and is_logger:
    wandb.finish()
