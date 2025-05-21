from typing import Optional, Literal
from zencfg import ConfigBase

class OptimizationConfig(ConfigBase):
    n_epochs: int
    training_loss: Literal['h1', 'l2'] = "h1"
    testing_loss: str = "l2"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    eval_interval: int = 1
    mixed_precision: bool = False
    scheduler: Literal['StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR'] = "StepLR"
    scheduler_T_max: int = 500
    scheduler_patience: int = 50
    step_size: int = 100
    gamma: float = 0.5

class PatchingConfig(ConfigBase):
    levels: int = 0
    padding: int = 0
    stitching: bool = False