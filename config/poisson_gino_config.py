from typing import Any, List, Optional, Dict

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .models import ModelConfig, GINO_Poisson2d
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

class NonlinearPoissonDatasetConfig(ConfigBase):
    file: str = "~/data/nonlin_poisson/nonlinear_poisson.obj"
    batch_size: int = 1
    test_batch_size: int = 1
    n_train: int = 7000
    n_test: int = 3000
    n_in: int = 5000
    n_out: int = 100
    n_eval: int = 6000
    n_bound: int = 4000
    query_resolution: int = 64
    train_out_res: int = 400
    padding: int = 1
    single_instance: bool = False
    input_min: int = 100
    input_max: int = 5000
    sample_random_in: Optional[Any] = None
    sample_random_out: Optional[Any] = None
    return_queries_dict: bool = True

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = GINO_Poisson2d()
    opt: OptimizationConfig = MGNOPoissonOptConfig()
    data: ConfigBase = NonlinearPoissonDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()
