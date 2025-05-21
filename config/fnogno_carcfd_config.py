from typing import Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .models import ModelConfig, FNOGNO_Small3d
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig

class CarCFDDatasetConfig(ConfigBase):
    root: str
    sdf_query_resolution: int
    n_train: int = 500
    n_test: int = 111
    download: bool = True

class CarCFDOptConfig(OptimizationConfig):
    n_epochs: int = 300,
    learning_rate: float = 5e-3,
    training_loss: str = "h1",
    weight_decay: float = 1e-4,
    scheduler: str = "StepLR",
    step_size: str = 60,
    gamma: str = 0.5,

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNOGNO_Small3d()
    opt: OptimizationConfig = OptimizationConfig()
    data: CarCFDDatasetConfig = CarCFDDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()