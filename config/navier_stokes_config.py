from typing import Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .models import ModelConfig, FNO_Medium2d
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig

class NavierStokesOptConfig(OptimizationConfig):
    n_epochs: int = 600
    learning_rate: float = 3e-4
    training_loss: str = "h1"
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    step_size: int = 100
    gamma: float = 0.5

class NavierStokesDatasetConfig(ConfigBase):
    folder: str = "~/data/navier_stokes/"
    batch_size: int = 8
    n_train: int = 10000
    train_resolution: int = 128
    n_tests: List[int] = [2000]
    test_resolutions: List[int] = [128]
    test_batch_sizes: List[int] = [8]
    encode_input: bool = True
    encode_output: bool = True

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNO_Medium2d()
    opt: OptimizationConfig = NavierStokesOptConfig()
    data: NavierStokesDatasetConfig = NavierStokesDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()
