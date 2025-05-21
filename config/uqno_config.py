from typing import Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .models import ModelConfig, FNO_Medium2d
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig

class NavierStokesOptConfig(OptimizationConfig):
    n_epochs: int = 600,
    learning_rate: float = 3e-4,
    training_loss: str = "h1",
    weight_decay: float = 1e-4,
    scheduler: str = "StepLR",
    step_size: int = 100,
    gamma: float = 0.5,

class Opt(ConfigBase):
    
    solution: Dict[str, Any] = {'n_epochs': 300, 'resume': False, 'learning_rate': '5e-3', 'training_loss': 'h1', 'weight_decay': '1e-4', 'mixed_precision': False, 'scheduler_T_max': 500, 'scheduler_patience': 5, 'scheduler': 'StepLR', 'step_size': 60, 'gamma': 0.5}
    residual: Dict[str, Any] = {'n_epochs': 300, 'learning_rate': '5e-3', 'training_loss': 'h1', 'weight_decay': '1e-4', 'mixed_precision': False, 'scheduler_T_max': 500, 'scheduler_patience': 5, 'scheduler': 'StepLR', 'step_size': 60, 'gamma': 0.5}


class SolutionModelOptConfig(OptimizationConfig):
    n_epochs: int = 300
    resume: bool = False
    learning_rate: float = 5e-3
    training_loss: str = "h1"
    weight_decay: float = 1e-4
    mixed_precision: bool = False
    scheduler_T_max: int = 500
    scheduler_patience: int = 5
    scheduler: str = "StepLR"
    step_size: int = 60
    gamma: float = 0.5

class ResidualModelOptConfig(OptimizationConfig):
    n_epochs: int = 300
    learning_rate: float = 5e-3
    training_loss: str = "h1"
    weight_decay: float = 1e-4
    mixed_precision: bool = False
    scheduler_T_max: int = 500
    scheduler_patience: int = 5
    scheduler: str = "StepLR"
    step_size: int = 60
    gamma: float = 0.5

class UQNODarcyDatasetConfig(ConfigBase):
    root: str = "YOUR_ROOT"
    batch_size: int = 4
    n_train_total: int = 4000
    n_train_solution: int = 2500
    n_train_residual: int = 1000
    n_calib_residual: int = 500
    train_resolution: int = 421
    n_test: int = 1000
    test_resolution: int = 421
    test_batch_size: int = 4
    encode_input: bool = True
    encode_output: bool = True

class UQNO_OptConfig(ConfigBase):
    alpha: float = 0.9
    delta: float = 0.95
    solution: SolutionModelOptConfig = SolutionModelOptConfig()
    residual: ResidualModelOptConfig = ResidualModelOptConfig()

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNO_Medium2d()
    opt: OptimizationConfig = UQNO_OptConfig()
    data: UQNODarcyDatasetConfig = UQNODarcyDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()
