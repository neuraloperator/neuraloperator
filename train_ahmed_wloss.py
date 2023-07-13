import matplotlib
matplotlib.use("Agg")  # Set the backend to Agg
import os
from typing import List, Tuple, Union
import numpy as np
import yaml
from timeit import default_timer
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import wandb
import sys
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from neuralop.training.loggers import init_logger
from neuralop.utils import get_wandb_api_key, count_params
from neuralop.datasets.mesh_datamodule import MeshDataModule
from neuralop.training.losses import total_drag, IregularLpqLoss, LpLoss
from neuralop.models.FNOGNO import FNOGNO
from neuralop.training.average_meter import AverageMeter, AverageMeterDict

def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def str2intlist(s: str) -> List[int]:
    return [int(item.strip()) for item in s.split(",")]

class DotDict(dict):
    """
    dot.notation access to dictionary attributes
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"


def flatten_dict(d, parent_key="", sep="_", no_sep_keys=["base"]):
    items = []
    for k, v in d.items():
        # Do not expand parent key if it is "base"
        if parent_key in no_sep_keys:
            new_key = k
        else:
            new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def instantiate_scheduler(optimizer, config):
    if config.opt_scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.opt_scheduler_T_max
        )
    elif config.opt_scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.opt_step_size, gamma=config.opt_gamma
        )
    else:
        raise ValueError(f"Got {config.opt.scheduler=}")
    return scheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/FNOInterpAhmed.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (cuda or cpu)",
    )
    parser.add_argument(
        "--data_path", type=str, default=None, help="Override data_path in config file"
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file to resume training",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="log",
        help="Path to the log directory",
    )
    parser.add_argument("--logger_types", type=str, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for training")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument(
        "--sdf_spatial_resolution",
        type=str2intlist,
        default=None,
        help="SDF spatial resolution. Use comma to separate the values e.g. 32,32,32.",
    )

    args = parser.parse_args()
    return args

def load_config(config_path):
    def include_constructor(loader, node):
        # Get the path of the current YAML file
        current_file_path = loader.name

        # Get the folder containing the current YAML file
        base_folder = os.path.dirname(current_file_path)

        # Get the included file path, relative to the current file
        included_file = os.path.join(base_folder, loader.construct_scalar(node))

        # Read and parse the included file
        with open(included_file, "r") as file:
            return yaml.load(file, Loader=yaml.Loader)

    # Register the custom constructor for !include
    yaml.Loader.add_constructor("!include", include_constructor)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Convert to dot dict
    config_flat = flatten_dict(config)
    config_flat = DotDict(config_flat)
    return config_flat

def data_dict_to_input(model, data_dict, device="cuda:0"):
    x_in = data_dict["centroids"][0]  # (n_in, 3)
    x_out = data_dict["query_points"][0] # (n_x, n_y, n_z, 3)
    df = data_dict["distance"]  # (1, n_x, n_y, n_z)

    info_fields = data_dict['inlet_velocity'] * torch.ones_like(df)

    df = torch.cat((df, info_fields), dim=0)

    if model.use_adain:
        vel = torch.tensor([data_dict['inlet_velocity']]).view(-1, ).cuda()
        vel_embed = model.adain_pos_embed(vel)
        for norm in model.fno.fno_blocks.norm:
            norm.update_embeddding(vel_embed)

    x_in, x_out, df = (
        x_in.to(device),
        x_out.to(device),
        df.to(device),
    )

    return x_in, x_out, df

def cal_loss_dict(model, data_dict, loss_fn, max_in_points=None, device="cuda:0"):
    x_in, x_out, df = data_dict_to_input(model, data_dict)

    if max_in_points is not None:
        r = min(max_in_points, x_in.shape[0])
        indices = torch.randperm(x_in.shape[0])[:r]
        x_in = x_in[indices, ...]

    pred = model(x_in, x_out, df)

    if max_in_points is not None:
        truth = data_dict["pressure"][0][indices].view(1, -1).to(device)
    else:
        truth = data_dict["pressure"][0].view(1, -1).to(device)

    return {"loss": loss_fn(pred.view(1, -1), truth, data_dict["triangle_areas"].to(device))}


@torch.no_grad()
def eval_dict(model, data_dict, loss_fn, decode_fn=None, max_in_points=None, device="cuda:0"):
    x_in, x_out, df = data_dict_to_input(model, data_dict)

    if max_in_points is not None:
        r = min(max_in_points, x_in.shape[0])
        pred_chunks = []
        x_in_chunks = torch.split(x_in, r, dim=0)
        for j in range(len(x_in_chunks)):
            pred_chunks.append(model(x_in_chunks[j], x_out, df))
        pred = torch.cat(tuple(pred_chunks), dim=0)
    else:
        pred = model(x_in, x_out, df)

    pred = pred.reshape(1, -1)
    truth = data_dict["pressure"].to(device).reshape(1, -1)
    out_dict = {"l2": loss_fn(pred, truth, data_dict["triangle_areas"].to(device))}
    if decode_fn is not None:
        pred = decode_fn(pred.cpu()).cuda()
        truth = decode_fn(truth.cpu()).cuda()
        out_dict["l2_decoded"] = loss_fn(pred, truth, data_dict["triangle_areas"].to(device))
    return out_dict


@torch.no_grad()
def eval(model, datamodule, config, loss_fn):
    model.eval()
    test_loader = datamodule.test_dataloader(
        batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    eval_meter = AverageMeterDict()
    visualize_data_dicts = []
    for i, data_dict in enumerate(test_loader):
        out_dict = eval_dict(
            model, data_dict, loss_fn=loss_fn, decode_fn=datamodule.normalizers['pressure'].decode, max_in_points=config.max_in_points
        )
        eval_meter.update(out_dict)
        if i % config.test_plot_interval == 0:
            visualize_data_dicts.append(data_dict)

    # Merge all dictionaries
    merged_image_dict = {}
    if hasattr(model, "image_dict"):
        for i, data_dict in enumerate(visualize_data_dicts):
            image_dict = model.image_dict(data_dict)
            for k, v in image_dict.items():
                merged_image_dict[f"{k}_{i}"] = v

    model.train()

    return eval_meter.avg, merged_image_dict


def train(config, device: Union[torch.device, str] = "cuda:0"):
    device = torch.device(device)
    model = FNOGNO().to(device)
    #Load ahmed body
    datamodule = MeshDataModule('data/new_ahmed/new_ahmed', 'case', 
                          query_points=[64,64,64], 
                          n_train=5, 
                          n_test=5, 
                          attributes=['pressure', 'wall_shear_stress', 'inlet_velocity', 'info', 'drag_history'])
    train_loader = datamodule.train_dataloader(batch_size=config.batch_size, shuffle=True)
    
    loggers = init_logger(config)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = instantiate_scheduler(optimizer, config)

    # Initialize the loss function
    loss_fn = IregularLpqLoss()
    for ep in range(config.num_epochs):
        model.train()
        t1 = default_timer()
        train_l2_meter = AverageMeter()
        # train_reg = 0
        for data_dict in train_loader:
            optimizer.zero_grad()
            loss_dict = cal_loss_dict(model, data_dict, loss_fn=loss_fn, max_in_points=config.max_in_points)
            loss = 0
            for k, v in loss_dict.items():
                loss = loss + v.mean()
            loss.backward()

            optimizer.step()

            train_l2_meter.update(loss.item())

            loggers.log_scalar("train/lr", scheduler.get_lr()[0], ep)
            loggers.log_scalar("train/loss", loss.item(), ep)

        scheduler.step()
        t2 = default_timer()
        print(
            f"Training epoch {ep} took {t2 - t1:.2f} seconds. L2 loss: {train_l2_meter.avg:.4f}"
        )
        loggers.log_scalar("train/train_l2", train_l2_meter.avg, ep)
        loggers.log_scalar("train/train_epoch_duration", t2 - t1, ep)

        if ep % config.eval_interval == 0 or ep == config.num_epochs - 1:
            eval_dict, eval_images = eval(model, datamodule, config, loss_fn)
            for k, v in eval_dict.items():
                print(f"Epoch: {ep} {k}: {v:.4f}")
                loggers.log_scalar(f"eval/{k}", v, ep)
            for k, v in eval_images.items():
                loggers.log_image(f"eval/{k}", v, ep)

        torch.save(model.state_dict(), config.model_path)

if __name__ == "__main__":
    args = parse_args()
    # print command line args
    print(args)
    config = load_config(args.config)

    # Update config with command line arguments
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value

    # pretty print the config
    for key, value in config.items():
        print(f"{key}: {value}")

    # Set the random seed
    if config.seed is not None:
        set_seed(config.seed)
    train(config, device=args.device)
