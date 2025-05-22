from typing import Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .models import ModelConfig, FNO_Small2d
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig


class DarcyOptConfig(OptimizationConfig):
    n_epochs: int = 300
    learning_rate: float = 5e-3
    training_loss: str = "h1"
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    step_size: int = 60
    gamma: float = 0.5

class DarcyDatasetConfig(ConfigBase):
    folder: str = "~/data/darcy/"
    batch_size: int = 8
    n_train: int = 1000
    train_resolution: int = 16
    n_tests: List[int] = [100, 50]
    test_resolutions: List[int] = [16, 32]
    test_batch_sizes: List[int] = [16, 16]
    encode_input: bool = True
    encode_output: bool = True
    download: bool = True

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "fno"
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNO_Small2d()
    opt: OptimizationConfig = DarcyOptConfig()
    data: DarcyDatasetConfig = DarcyDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()