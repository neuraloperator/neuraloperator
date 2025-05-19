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

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "fno"
    distributed: DistributedConfig = DistributedConfig()
    fno: FNOConfig = Small2dFNO()
    '''
    OR
    fno: FNOConfig = FNOConfig(
        data_channels = 1
        out_channels = 1
        n_modes = [16,16]
        hidden_channels = 24
        projection_channel_ratio = 2
    )
    '''
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