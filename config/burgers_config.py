from typing import Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .datasets import BurgersDatasetConfig
from .models import FNOConfig, Small2dFNOConfig
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "fno"
    distributed: DistributedConfig = DistributedConfig()
    fno: FNOConfig = Small2dFNOConfig()
    opt: OptimizationConfig = OptimizationConfig(
        n_epochs=10000
    )
    data: BurgersDatasetConfig = BurgersDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig(
        log=True,
        entity="my_entity",
        project="my_project"
    )