from typing import Any, List, Optional

from zencfg import ConfigBase
from ..distributed import DistributedConfig
from ..models import ModelConfig, FNO_Medium3d
from ..opt import OptimizationConfig, PatchingConfig
from ..wandb import WandbConfig

class MHD64OptConfig(OptimizationConfig):
    n_epochs: int = 600
    learning_rate: float = 3e-4
    training_loss: str = "h1"
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    step_size: int = 100
    gamma: float = 0.5

class MHD_64_DatasetConfig(ConfigBase):
    root: str = "~/data/the_well"
    n_train: int = 10
    n_steps_input: int = 1
    n_steps_output: int = 1
    batch_size: int = 1
    n_test: int = 10 
    test_batch_size: int = 1 
    input_timesteps: int = 1
    output_timesteps: int = 51 #15
    max_rollout_len: int = 100

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNO_Medium3d(data_channels=7,
                                      out_channels=7,)
    opt: OptimizationConfig = MHD64OptConfig()
    data: MHD_64_DatasetConfig = MHD_64_DatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()
