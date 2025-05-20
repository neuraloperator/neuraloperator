from typing import Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .datasets import CarCFDDatasetConfig
from .models import FNOGNOConfig, CarCFDFNOGNOConfig
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "fno"
    distributed: DistributedConfig = DistributedConfig()
    fno: FNOGNOConfig = CarCFDFNOGNOConfig()
    opt: OptimizationConfig = OptimizationConfig(
        n_epochs=300,
        learning_rate=5e-3,
        training_loss="h1",
        weight_decay=1e-4,
        scheduler="StepLR",
        step_size=60,
        gamma=0.5,
    )
    data: CarCFDDatasetConfig = CarCFDDatasetConfig(
        root="/home/YOURNAME/data/car-pressure-data/",
        sdf_query_resolution=32,
        n_train=500,
        n_test=111,
        download=True,
    )
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig(
        log=False, # turn this to True to log to wandb
        entity="my_entity",
        project="my_project"
    )