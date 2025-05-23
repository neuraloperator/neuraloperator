from typing import Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .datasets import NavierStokesDatasetConfig
from .models import ModelConfig, SimpleFNOConfig
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig

class TestConfig(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = SimpleFNOConfig(
        data_channels=1,
        out_channels=1,
        n_modes=[64,64],
        hidden_channels=64,
        n_layers=4,
        projection_channel_ratio=4,
    )
    opt: OptimizationConfig = OptimizationConfig(
        n_epochs=600,
        learning_rate=3e-4,
        training_loss="h1",
        weight_decay=1e-4,
        scheduler="StepLR",
        step_size=100,
        gamma=0.5,
    )
    data: NavierStokesDatasetConfig = NavierStokesDatasetConfig(
        batch_size=8,
        n_train=10000,
        train_resolution=128,
        n_tests=[2000],
        test_resolutions=[128],
        test_batch_sizes=[8],
        encode_input=True,
        encode_output=True,
    )
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig(
        log=False, # turn this to True to log to wandb
        entity="my_entity",
        project="my_project"
    )

