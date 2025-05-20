from typing import Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .datasets import DarcyDatasetConfig
from .models import FNOConfig, Small2dFNOConfig
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "fno"
    distributed: DistributedConfig = DistributedConfig()
    fno: FNOConfig = Small2dFNOConfig(
        hidden_channels = 32,
        norm = "group_norm"
    )
    opt: OptimizationConfig = OptimizationConfig(
        n_epochs=300,
        learning_rate=5e-3,
        training_loss="h1",
        weight_decay=1e-4,
        scheduler="StepLR",
        step_size=60,
        gamma=0.5,
    )
    data: DarcyDatasetConfig = DarcyDatasetConfig(
        batch_size=16,
        n_train=1000,
        train_resolution=16,
        n_tests=[100, 50],
        test_resolutions=[16, 32],
        test_batch_sizes=[16, 16],
        encode_input=True,
        encode_output=True,
    )
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig(
        log=False, # turn this to True to log to wandb
        entity="my_entity",
        project="my_project"
    )