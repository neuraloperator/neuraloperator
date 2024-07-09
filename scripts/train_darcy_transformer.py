import sys

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from neuralop import get_model
from neuralop.losses import H1Loss, LpLoss
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.layers.embeddings import GridEmbedding2D
from neuralop.training import Trainer, setup
from neuralop.training.callbacks import BasicLoggerCallback
from neuralop.utils import get_wandb_api_key, count_model_params


# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./darcy_transformer_config.yaml", config_name="default", config_folder="../config"
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
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.transformer_no.encoder_n_layers,
                config.transformer_no.encoder_hidden_channels,
                config.transformer_no.decoder_hidden_channels,
                config.transformer_no.encoder_num_heads,
                config.transformer_no.decoder_num_heads,
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

# Loading the Darcy flow dataset
train_loader, test_loaders, default_data_processor = load_darcy_flow_small(
    data_root="../neuralop/data/datasets/data/",
    n_train=config.data.n_train,
    batch_size=config.data.batch_size,
    positional_encoding=config.data.positional_encoding,
    test_resolutions=config.data.test_resolutions,
    n_tests=config.data.n_tests,
    test_batch_sizes=config.data.test_batch_sizes,
    encode_input=config.data.encode_input,
    encode_output=config.data.encode_output,
)

model = get_model(config)

class TransformerNODataProcessor(DefaultDataProcessor):
    """
    TransformerNODataProcessor provides almost the same functionality as a 
    DefaultDataProcessor but splits the input into two tensors `(u, pos_src)`
    so that some model forward operations can operate on only the input function `u`
    and some can operate on both `u` and the positional embedding `pos_src`
    """
    def __init__(self, in_normalizer=None, out_normalizer=None, device='cpu'):
        super().__init__()
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer
        pos_embed = GridEmbedding2D(grid_boundaries=[[0,1], [0,1]])
        self.positional_embedding = pos_embed
        self.device = device
    
    def preprocess(self, data_dict):
        x = data_dict["x"]
        batch_size = x.shape[0]
        grid_res = x.shape[2:4]
        if self.in_normalizer:
            x = self.in_normalizer.transform(x)
        
        pos = self.positional_embedding.coords_only(grid_res)
        y = data_dict["y"]
        if self.out_normalizer and self.training:
            y = self.out_normalizer.transform(y)
       
        u = x.permute(0,2,3,1).to(self.device).view(batch_size, -1, 1)
        pos_src = pos.permute(1,2,0).to(self.device).view(-1, 2).unsqueeze(0)
        data_dict["x"] = u
        data_dict["pos_src"] = pos_src
        data_dict["y"] = y.permute(0,2,3,1).to(self.device)

        return data_dict
    
    def postprocess(self, out, data_dict):
        y = data_dict["y"]
        batch_size, _, nx, ny = y.shape
        out = out.view(batch_size, nx, ny, -1).permute(0, 3, 1, 2)
        if self.out_normalizer and not self.training:
            out = self.out_normalizer.inverse_transform(out)
            y = self.out_normalizer.inverse_transform(y)
        data_dict["y"] = y

        return out, data_dict

data_processor = TransformerNODataProcessor(in_normalizer=default_data_processor.in_normalizer,
                                            out_normalizer=default_data_processor.out_normalizer).to(device)
model = model.to(device)


# Use distributed data parallel
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


# Creating the losses
l2loss = LpLoss(d=3, p=2)
h1loss = H1Loss(d=3)
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

if config.wandb.log:
    logger = BasicLoggerCallback(**wandb_args)
else:
    logger = BasicLoggerCallback()

trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    data_processor=data_processor,
    device=device,
    amp_autocast=config.opt.amp_autocast,
    wandb_log=config.wandb.log,
    log_test_interval=config.wandb.log_test_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose and is_logger,
    callbacks=[
        logger
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

if config.wandb.log and is_logger:
    wandb.finish()
