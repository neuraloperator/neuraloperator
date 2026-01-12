from typing import Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .models import ModelConfig, OTNO_Small2d
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig


class CarCFDDatasetConfig(ConfigBase):
    root: str = "~/processed-car-pressure-data"
    n_train: int = 500
    n_test: int = 111
    reg: float = 1e-06
    expand_factor: float = 3.0


class CarCFDOptConfig(OptimizationConfig):
    n_epochs: int = 100
    learning_rate: float = 1e-3
    training_loss: str = "l2"
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    step_size: int = 30
    gamma: float = 0.5


class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = OTNO_Small2d()
    opt: OptimizationConfig = CarCFDOptConfig()
    data: CarCFDDatasetConfig = CarCFDDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()
