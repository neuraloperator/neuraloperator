from typing import Any, List, Optional

from zencfg import ConfigBase
from config.distributed import DistributedConfig
from config.models import ModelConfig, FNO_Small2d
from config.opt import OptimizationConfig, PatchingConfig
from config.wandb import WandbConfig

class Data(ConfigBase):
    train_resolution: int = 128
    batch_size: int = 2

class TestConfig(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNO_Small2d()
    data: Data = Data()
    opt: OptimizationConfig = OptimizationConfig(
        n_epochs=10
    )
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()

