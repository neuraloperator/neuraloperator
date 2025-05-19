from typing import Any, List, Optional


from zencfg import ConfigBase
from .distributed import DistributedConfig
from .datasets import BurgersDatasetConfig
from .models import FNOConfig
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig

class Small2dFNO(FNOConfig):
    data_channels: int = 1
    out_channels: int = 1
    n_modes: List[int] = [16,16]
    hidden_channels: int = 24
    projection_channel_ratio: int = 2

class BurgersOptConfig(OptimizationConfig):
    n_epochs = 10000

class BurgersWandbConfig(WandbConfig):
    log = False
    entity = "my_entity"
    project = "my_project"
    log_outputs = False

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "fno"
    distributed: DistributedConfig = DistributedConfig()
    fno: FNOConfig = Small2dFNO()
    opt: OptimizationConfig = BurgersOptConfig()
    data: BurgersDatasetConfig = BurgersDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = BurgersWandbConfig()