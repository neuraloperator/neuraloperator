"""
Training script for UQNO (Uncertainty Quantification Neural Operator) on Darcy flow.

This script implements a two-stage training process:
1. Train a base solution model on Darcy flow data
2. Train a residual model to predict uncertainty bounds
3. Calibrate the uncertainty predictions for reliable coverage

The UQNO approach provides both point predictions and uncertainty estimates
for neural operator predictions on PDEs.
"""

import sys
import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.data.datasets.darcy import DarcyDataset
from neuralop.data.datasets.tensor_dataset import TensorDataset
from neuralop.data.transforms.data_processors import DataProcessor, DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.losses.data_losses import PointwiseQuantileLoss
from neuralop.models import UQNO
from neuralop.training import setup
from neuralop.utils import get_wandb_api_key, count_model_params

# Configuration setup
config_name = "default"
from zencfg import make_config_from_cli
import sys

sys.path.insert(0, "../")
from config.uqno_config import Default

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
if config.verbose and is_logger and config.opt.solution.n_epochs > 0:
    print(f"##### CONFIG #####\n{config}")
    sys.stdout.flush()

# Loading the Darcy flow dataset for training the base model
root_dir = Path(config.data.root).expanduser()
solution_dataset = DarcyDataset(
    root_dir=root_dir,
    n_train=config.data.n_train_total,
    n_tests=[config.data.n_test],
    batch_size=config.data.batch_size,
    test_batch_sizes=[config.data.test_batch_size],
    train_resolution=421,
    test_resolutions=[421],
    encode_input=config.data.encode_input,
    encode_output=config.data.encode_output,
)

train_db = solution_dataset.train_db

test_db = solution_dataset.test_dbs[421]
test_loaders = {
    421: DataLoader(
        test_db,
        shuffle=False,
        batch_size=config.data.test_batch_size,
    )
}
data_processor = solution_dataset.data_processor

# split the training set up into train, residual_train, residual_calibration

solution_train_db = TensorDataset(**train_db[: config.data.n_train_solution])
residual_train_db = TensorDataset(
    **train_db[
        config.data.n_train_solution : config.data.n_train_solution
        + config.data.n_train_residual
    ]
)
residual_calib_db = TensorDataset(
    **train_db[
        config.data.n_train_solution
        + config.data.n_train_residual : config.data.n_train_solution
        + config.data.n_train_residual
        + config.data.n_calib_residual
    ]
)

data_processor = data_processor.to(device)

# Base solution model initialization
solution_model = get_model(config)
solution_model = solution_model.to(device)

# Create the optimizer
optimizer = torch.optim.Adam(
    solution_model.parameters(),
    lr=config.opt.solution.learning_rate,
    weight_decay=config.opt.solution.weight_decay,
)

if config.opt.solution.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.solution.gamma,
        patience=config.opt.solution.scheduler_patience,
        mode="min",
    )
elif config.opt.solution.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.solution.scheduler_T_max
    )
elif config.opt.solution.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.opt.solution.step_size,
        gamma=config.opt.solution.gamma,
    )
else:
    raise ValueError(f"Got scheduler={config.opt.solution.scheduler}")


# Loss function configuration for base model
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
if config.opt.solution.training_loss == "l2":
    train_loss = l2loss
elif config.opt.solution.training_loss == "h1":
    train_loss = h1loss
else:
    raise ValueError(
        f"Got training_loss={config.opt.solution.training_loss}"
        f'but expected one of ["l2", "h1"]'
    )
eval_losses = {"h1": h1loss, "l2": l2loss}

# if not config.load_soln_model:
if config.verbose and is_logger and config.opt.solution.n_epochs > 0:
    print("\n### MODEL ###\n", solution_model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()


# Log model parameter count
if is_logger:
    n_params = count_model_params(solution_model)

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


solution_train_loader = DataLoader(
    solution_train_db,
    batch_size=config.data.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=False,
)

trainer = Trainer(
    model=solution_model,
    n_epochs=config.opt.solution.n_epochs,
    device=device,
    data_processor=data_processor,
    mixed_precision=config.opt.solution.mixed_precision,
    wandb_log=config.wandb.log,
    eval_interval=config.opt.solution.eval_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose and is_logger,
)
if config.opt.solution.n_epochs > 0:
    if config.opt.solution.resume == True:
        resume_dir = "./solution_ckpts"
    else:
        resume_dir = None

    # save the best solution model

    trainer.train(
        train_loader=solution_train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
        save_best="421_l2",
        save_dir="./solution_ckpts",
        resume_from_dir=resume_dir,
    )

#############################
# UQ Training and Utilities #
#############################


def loader_to_residual_db(model, data_processor, loader, device, train_val_split=True):
    """
    Convert a dataset of solution data to residual data for uncertainty training.

    This function takes a dataset of input-output pairs (x: a(x), y: u(x)) and
    converts it to residual data (x: a(x), y: G(a,x) - u(x)) where G is the
    trained solution model. This residual data is used to train the uncertainty
    quantification model.

    Parameters
    ----------
    model : nn.Module
        Trained solution model (frozen) that predicts G(a,x)
    data_processor : DataProcessor
        Data processor used to train the solution model
    loader : DataLoader
        Data loader containing solution data to convert to residuals.
        Must be drawn from the same distribution as the solution model's
        training distribution
    device : str or torch.device
        Device to run computations on
    train_val_split : bool, optional
        Whether to split the residual data into training and validation sets,
        by default True

    Returns
    -------
    tuple
        (residual_train_db, residual_val_db, residual_data_processor)
        - residual_train_db: Training dataset of residuals
        - residual_val_db: Validation dataset of residuals (or None if train_val_split=False)
        - residual_data_processor: Data processor for residual data
    """
    error_list = []
    x_list = []
    model = model.to(device)
    model.eval()
    data_processor.eval()  # unnormalized y
    data_processor = data_processor.to(device)
    for idx, sample in enumerate(loader):
        sample = data_processor.preprocess(sample)
        out = model(**sample)
        out, sample = data_processor.postprocess(out, sample)  # unnormalize output

        x_list.append(sample["x"].to("cpu"))
        error = (
            (out - sample["y"]).detach().to("cpu")
        )  # detach, otherwise residual carries gradient of model weight
        # error is unnormalized here
        error_list.append(error)

        del sample, out
    errors = torch.cat(error_list, axis=0)
    xs = torch.cat(x_list, axis=0)  # check this

    residual_encoder = UnitGaussianNormalizer()
    residual_encoder.fit(errors)

    # positional encoding and normalization already applied to X values
    residual_data_processor = DefaultDataProcessor(
        in_normalizer=None, out_normalizer=residual_encoder
    )
    residual_data_processor.train()

    if train_val_split:
        val_start = int(0.8 * xs.shape[0])

        residual_train_db = TensorDataset(x=xs[:val_start], y=errors[:val_start])
        residual_val_db = TensorDataset(x=xs[val_start:], y=errors[val_start:])
    else:
        residual_val_db = None
    return residual_train_db, residual_val_db, residual_data_processor


class UQNODataProcessor(DataProcessor):
    """
    Data processor for UQNO (Uncertainty Quantification Neural Operator) training.

    This processor handles the conversion of UQNO model outputs and ground truth
    into the format expected by PointwiseQuantileLoss for uncertainty training.
    It converts the tuple (G_hat(a,x), E(a,x)) and ground truth G_true(a,x) into:
    - y_pred = E(a,x) (uncertainty predictions)
    - y_true = abs(G_hat(a,x) - G_true(a,x)) (absolute prediction errors)

    The processor also preserves all necessary transformations for both the
    base solution model and residual model inputs/outputs.
    """

    def __init__(
        self,
        base_data_processor: DataProcessor,
        resid_data_processor: DataProcessor,
        device: str = "cpu",
    ):
        """
        Initialize the UQNO data processor.

        Parameters
        ----------
        base_data_processor : DataProcessor
            Data processor for base solution model input/output transformations
        resid_data_processor : DataProcessor
            Data processor for residual model input/output transformations
        device : str, optional
            Device to run computations on ("cpu" or "cuda"), by default "cpu"
        """
        super().__init__()
        self.base_data_processor = base_data_processor
        self.residual_normalizer = resid_data_processor.out_normalizer

        self.device = device
        self.scale_factor = None

    def set_scale_factor(self, factor):
        """
        Set the uncertainty scaling factor for calibration.

        Parameters
        ----------
        factor : torch.Tensor
            Scaling factor to apply to uncertainty predictions
        """
        self.scale_factor = factor.to(device)

    def wrap(self, model):
        """
        Wrap the model with this data processor.

        Parameters
        ----------
        model : nn.Module
            Model to wrap with this data processor

        Returns
        -------
        self
            Returns self for method chaining
        """
        self.model = model
        return self

    def to(self, device):
        """
        Move the data processor to the specified device.

        Parameters
        ----------
        device : str or torch.device
            Device to move the processor to

        Returns
        -------
        self
            Returns self for method chaining
        """
        self.device = device
        self.base_data_processor = self.base_data_processor.to(device)
        self.residual_normalizer = self.residual_normalizer.to(device)
        return self

    def train(self):
        """Set the data processor to training mode."""
        self.base_data_processor.train()

    def eval(self):
        """Set the data processor to evaluation mode."""
        self.base_data_processor.eval()

    def preprocess(self, *args, **kwargs):
        """
        Preprocess input data using the base data processor.

        No additional preprocessing is required - this method simply
        wraps the base data processor's preprocessing functionality.
        """
        return self.base_data_processor.preprocess(*args, **kwargs)

    def postprocess(self, out, sample):
        """
        Postprocess UQNO model outputs and ground truth data.

        This method handles the conversion of UQNO outputs (G_hat, E) and
        ground truth into the format expected by PointwiseQuantileLoss.
        It also applies inverse normalization to both predictions and ground truth.
        """
        self.base_data_processor.eval()
        g_hat, pred_uncertainty = out  # UQNO returns a tuple

        pred_uncertainty = self.residual_normalizer.inverse_transform(pred_uncertainty)
        # this is normalized

        g_hat, sample = self.base_data_processor.postprocess(
            g_hat, sample
        )  # unnormalize g_hat

        g_true = sample["y"]  # this is unnormalized in eval mode
        sample["y"] = g_true - g_hat  # both unnormalized

        sample.pop("x")  # remove x arg to avoid overloading loss args

        if self.scale_factor is not None:
            pred_uncertainty = pred_uncertainty * self.scale_factor
        return pred_uncertainty, sample

    def forward(self, **sample):
        """
        Complete forward pass through the UQNO data processor and model.

        This method combines preprocessing, model forward pass, and postprocessing
        for the UQNO training pipeline.
        """
        sample = self.preprocess(sample)
        out = self.model(**sample)
        out, sample = self.postprocess(out, sample)
        return out, sample


# Load the best-performing solution model from checkpoint
solution_model = solution_model.from_checkpoint(
    save_folder="./solution_ckpts", save_name="best_model_815"
)
solution_model = solution_model.to(device)

# Evaluate the solution model performance
eval_metrics = trainer.evaluate(eval_losses, data_loader=test_loaders[421], epoch=1)
print(f"Eval metrics = {eval_metrics}")

# Create residual model as a copy of the solution model
residual_model = copy.deepcopy(solution_model)
residual_model = residual_model.to(device)

# Set up quantile loss for uncertainty training
quantile_loss = PointwiseQuantileLoss(alpha= 1 - config.opt.alpha)

# Create the quantile model's optimizer
residual_optimizer = torch.optim.Adam(
    residual_model.parameters(),
    lr=config.opt.residual.learning_rate,
    weight_decay=config.opt.residual.weight_decay,
)


# Update WandB run name for uncertainty quantification phase
if wandb_args is not None:
    uq_wandb_name = "uq_" + wandb_args["name"]
    wandb_args["name"] = uq_wandb_name

# Residual model training phase
# Create data loader for residual training
residual_train_loader_unprocessed = DataLoader(
    residual_train_db,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=False,
)

# Convert solution data to residual data for uncertainty training
# This creates dataset of x: a(x), y: G_hat(a,x) - u(x)
(
    processed_residual_train_db,
    processed_residual_val_db,
    residual_data_processor,
) = loader_to_residual_db(
    solution_model, data_processor, residual_train_loader_unprocessed, device
)

residual_data_processor = residual_data_processor.to(device)

# Create data loaders for residual model training
residual_train_loader = DataLoader(
    processed_residual_train_db,
    batch_size=config.data.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=False,
)
residual_val_loader = DataLoader(
    processed_residual_val_db,
    batch_size=config.data.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=False,
)

# Configure learning rate scheduler for residual model
if config.opt.residual.scheduler == "ReduceLROnPlateau":
    resid_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        residual_optimizer,
        factor=config.opt.residual.gamma,
        patience=config.opt.residual.scheduler_patience,
        mode="min",
    )
elif config.opt.residual.scheduler == "CosineAnnealingLR":
    resid_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        residual_optimizer, T_max=config.opt.residual.scheduler_T_max
    )
elif config.opt.residual.scheduler == "StepLR":
    resid_scheduler = torch.optim.lr_scheduler.StepLR(
        residual_optimizer,
        step_size=config.opt.solution.step_size,
        gamma=config.opt.solution.gamma,
    )
else:
    raise ValueError(f"Got residual scheduler={config.opt.residual.scheduler}")

# Train the residual model if specified
if config.opt.residual.n_epochs > 0:
    # Create trainer for residual model
    residual_trainer = Trainer(
        model=residual_model,
        n_epochs=config.opt.residual.n_epochs,
        data_processor=residual_data_processor,
        wandb_log=config.wandb.log,
        device=device,
        mixed_precision=config.opt.residual.mixed_precision,
        eval_interval=config.opt.residual.eval_interval,
        log_output=config.wandb.log_output,
        use_distributed=config.distributed.use_distributed,
        verbose=config.verbose and is_logger,
    )

    # Train the residual model for uncertainty quantification
    residual_trainer.train(
        train_loader=residual_train_loader,
        test_loaders={"test": residual_val_loader},
        optimizer=residual_optimizer,
        scheduler=resid_scheduler,
        regularizer=False,
        training_loss=quantile_loss,
        eval_losses={"quantile": quantile_loss, "l2": l2loss},
        save_best="test_quantile",
        save_dir="./residual_ckpts",
    )

# Load the best residual model from checkpoint
residual_model = residual_model.from_checkpoint(
    save_name="best_model", save_folder="./residual_ckpts"
)
residual_model = residual_model.to(device)


# Uncertainty calibration phase
def get_coeff_quantile_idx(alpha, delta, n_samples, n_gridpts):
    """
    Compute quantile indices for uncertainty calibration.

    This function calculates the indices needed for uncertainty calibration
    based on the desired coverage parameters. It determines both domain-level
    and function-level quantile indices for reliable uncertainty estimation.

    Parameters
    ----------
    alpha : float
        Desired pointwise coverage level (percentage of points in domain)
    delta : float
        Desired function-level coverage (percentage of functions that satisfy alpha threshold)
    n_samples : int
        Number of function samples available for calibration
    n_gridpts : int
        Number of grid points per function (domain discretization)

    Returns
    -------
    tuple
        (domain_idx, function_idx)
        - domain_idx: Index for domain-level quantile selection
        - function_idx: Index for function-level quantile selection

    Notes
    -----
    The function computes quantile indices based on concentration inequalities.
    There is a minimum alpha that can be achieved based on the number of
    grid points, samples, and delta. The parameter t is chosen to balance
    domain-level and function-level coverage requirements.
    """
    lb = np.sqrt(-np.log(delta) / 2 / n_gridpts)
    t = (alpha - lb) / 3 + lb  # if t too small, will make the in-domain estimate conservative
    # too large will make the across-function estimate conservative. so we find a moderate t value
    print(f"we set alpha (on domain): {alpha}, t={t}")
    percentile = alpha - t
    domain_idx = int(np.ceil(percentile * n_gridpts))
    print(f"domain index: {domain_idx}'th largest of {n_gridpts}")

    # get function idx
    function_percentile = (
        np.ceil((n_samples + 1) * (delta - np.exp(-2 * n_gridpts * t * t))) / n_samples
    )
    function_idx = int(np.ceil(function_percentile * n_samples))
    print(f"function index: {function_idx}'th largest of {n_samples}")
    return domain_idx, function_idx


# Create the full UQNO model and data processor
uqno = UQNO(base_model=solution_model, residual_model=residual_model)
uqno_data_proc = UQNODataProcessor(
    base_data_processor=data_processor,
    resid_data_processor=residual_data_processor,
    device=device,
)

# Set to evaluation mode for calibration
uqno_data_proc.eval()

# Compute calibration ratios for uncertainty scaling
# This creates a list of (true error / uncertainty band) ratios
val_ratio_list = []
calib_loader = DataLoader(residual_calib_db, shuffle=True, batch_size=1)
with torch.no_grad():
    for idx, sample in enumerate(calib_loader):
        sample = uqno_data_proc.preprocess(sample)
        out = uqno(sample["x"])
        out, sample = uqno_data_proc.postprocess(out, sample)
        ratio = torch.abs(sample["y"]) / out
        val_ratio_list.append(ratio.squeeze().to("cpu"))
        del sample, out

# Stack all calibration ratios
val_ratios = torch.stack(val_ratio_list)

# Reshape for quantile computation
vr_view = val_ratios.view(val_ratios.shape[0], -1)


def eval_coverage_bandwidth(test_loader, alpha, device="cuda"):
    """
    Evaluate coverage and bandwidth of uncertainty predictions.

    This function computes the percentage of instances that achieve the target
    pointwise coverage level and calculates the average uncertainty bandwidth.
    It provides metrics for assessing the quality of uncertainty quantification.

    Parameters
    ----------
    test_loader : DataLoader
        Test data loader for evaluation
    alpha : float
        Target pointwise coverage level (e.g., 0.05 for 95% coverage)
    device : str, optional
        Device to run computations on, by default "cuda"

    Returns
    -------
    tuple
        (mean_interval, in_pred_percentage)
        - mean_interval: Average uncertainty bandwidth across test instances
        - in_pred_percentage: Percentage of instances achieving target coverage
    """
    in_pred_list = []
    avg_interval_list = []

    with torch.no_grad():
        for _, sample in enumerate(test_loader):
            sample = {k: v.to(device) for k, v in sample.items() if torch.is_tensor(v)}
            sample = uqno_data_proc.preprocess(sample)
            out = uqno(**sample)
            uncertainty_pred, sample = uqno_data_proc.postprocess(out, sample)
            pointwise_true_err = sample["y"]

            in_pred = (
                (torch.abs(pointwise_true_err) < torch.abs(uncertainty_pred))
                .float()
                .squeeze()
            )
            avg_interval = (
                torch.abs(uncertainty_pred.squeeze())
                .view(uncertainty_pred.shape[0], -1)
                .mean(dim=1)
            )
            avg_interval_list.append(avg_interval.to("cpu"))

            in_pred_flattened = in_pred.view(in_pred.shape[0], -1)
            in_pred_instancewise = (
                torch.mean(in_pred_flattened, dim=1) >= 1 - alpha
            )  # expected shape (batchsize, 1)
            in_pred_list.append(in_pred_instancewise.float().to("cpu"))

    in_pred = torch.cat(in_pred_list, axis=0)
    intervals = torch.cat(avg_interval_list, axis=0)
    mean_interval = torch.mean(intervals, dim=0)
    in_pred_percentage = torch.mean(in_pred, dim=0)
    print(
        f"{in_pred_percentage} of instances satisfy that >= {1-alpha} pts drawn are inside the predicted quantile"
    )
    print(f"Mean interval width is {mean_interval}")
    return mean_interval, in_pred_percentage


# Calibrate uncertainty predictions for different coverage levels
for alpha in [0.02, 0.05, 0.1]:
    for delta in [0.02, 0.05, 0.1]:
        # get quantile of domain gridpoints and quantile of function samples
        darcy_discretization = train_db[0]["x"].shape[-1] ** 2
        domain_idx, function_idx = get_coeff_quantile_idx(
            alpha, delta, n_samples=len(calib_loader), n_gridpts=darcy_discretization
        )

        # Compute uncertainty scaling factor based on calibration ratios
        val_ratios_pointwise_quantile = torch.topk(val_ratios.view(val_ratios.shape[0], -1),domain_idx+1, dim=1).values[:,-1]
        uncertainty_scaling_factor = torch.abs(torch.topk(val_ratios_pointwise_quantile, function_idx+1, dim=0).values[-1])
        print(f"scale factor: {uncertainty_scaling_factor}")

        # Apply scaling factor to uncertainty predictions
        uqno_data_proc.set_scale_factor(uncertainty_scaling_factor)

        # Evaluate coverage and bandwidth for this configuration
        uqno_data_proc.eval()
        print(f"------- for values {alpha=} {delta=} ----------")
        interval, percentage = eval_coverage_bandwidth(
            test_loader=test_loaders[train_db[0]["x"].shape[-1]],
            alpha=alpha,
            device=device,
        )

        # Log results to WandB
        if config.wandb.log and is_logger:
            wandb.log(interval, percentage)

# Finalize WandB logging
if config.wandb.log and is_logger:
    wandb.finish()
