from typing import Any, List, Optional, Dict

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .datasets import DataConfig, NonlinearPoissonDatasetConfig
from .models import ModelConfig, PoissonGINOConfig
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig


class MGNOPoissonOptConfig(OptimizationConfig):
    n_epochs: int = 1000
    training_loss: List[str] = ["equation", "boundary"]
    loss_weights: Dict[str, Any] = {'mse': 1.0, 'interior': '1e-2', 'boundary': 1.0}
    pino_method: str = "autograd"
    weight_decay: float = 1e-6
    mixed_precision: bool = False
    learning_rate: float = 1e-4
    optimizer: str = "Adam"
    scheduler: str = "ReduceLROnPlateau"
    scheduler_T_max: int = 5000
    scheduler_patience: int = 2
    gamma: float = 0.9

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = PoissonGINOConfig()
    opt: OptimizationConfig = MGNOPoissonOptConfig()
    data: DataConfig = NonlinearPoissonDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig(
        log=False, # turn this to True to log to wandb
        entity="my_entity",
        project="my_project"
    )
