from typing import Optional
from zencfg import ConfigBase

class OptimizationConfig(ConfigBase):
    n_epochs: int
    training_loss: str = "h1"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    eval_interval: int = 1
    amp_autocast: bool = False
    scheduler_T_max: int = 500
    scheduler_patience: int = 50
    scheduler: str = "StepLR"
    step_size: int = 100
    gamma: float = 0.5

class PatchingConfig(ConfigBase):
    levels: int = 0
    padding: int = 0
    stitching: bool = False