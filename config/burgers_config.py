from typing import Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .models import ModelConfig, FNO_Small2d
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig

class BurgersDatasetConfig(ConfigBase):
    folder: str = 'neuralop/data/datasets/data/' 
    batch_size: int = 16
    n_train: int = 800
    test_batch_sizes: List[int] = [16]
    n_tests: List[int] = [400]
    # full res is 128x101. We redistribute a mini version at 16x17
    spatial_length: int = 16 
    temporal_length: int = 17
    temporal_subsample: Optional[int] = None
    encode_input: bool = False
    encode_output: bool = False
    include_endpoint: List[bool] = [True, False]

class BurgersOptConfig(ConfigBase):
    n_epochs: int = 10000
    training_loss: str = "l2"
    testing_loss: str = "l2"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    eval_interval: int = 1
    mixed_precision: bool = False
    scheduler: str = 'ReduceLROnPlateau' # Or 'CosineAnnealingLR' OR 'ReduceLROnPlateau'
    scheduler_patience: int = 100 # For ReduceLROnPlateau only
    step_size: int = 60
    gamma: float = 0.5

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "fno"
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNO_Small2d()
    opt: BurgersOptConfig = BurgersOptConfig()
    data: BurgersDatasetConfig = BurgersDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()