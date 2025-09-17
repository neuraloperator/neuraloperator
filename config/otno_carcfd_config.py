from typing import Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .models import ModelConfig, OTNO_Small3d
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig

class CarCFDDatasetConfig(ConfigBase):
    root: str = "D:/python_code/data/car-pressure-data"
    sdf_query_resolution: int = 32
    n_train: int = 2#500
    n_test: int = 1#111
    reg: float = 1e-06
    expand_factor: float = 3.0


class CarCFDOptConfig(OptimizationConfig):
    n_epochs: int = 300
    learning_rate: float = 5e-3
    training_loss: str = "l2"
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    step_size: int = 60
    gamma: float = 0.5

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = OTNO_Small3d()
    opt: OptimizationConfig = CarCFDOptConfig()
    data: CarCFDDatasetConfig = CarCFDDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()
