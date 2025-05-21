from typing import Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .datasets import CarCFDDatasetConfig
from .models import ModelConfig, GINO_Small3d
from .opt import PatchingConfig
from .wandb import WandbConfig

class CarCFDOptConfig(ConfigBase):
    n_epochs: int = 301
    learning_rate: bool = True
    training_loss: str = "l2"
    testing_loss: str = "l2"
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    step_size: int = 50
    gamma: float = 0.5

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = GINO_Small3d()
    opt: ConfigBase = CarCFDOptConfig()
    data: CarCFDDatasetConfig = CarCFDDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig() # default empty