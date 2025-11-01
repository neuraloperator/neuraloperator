from typing import Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig
from .models import ModelConfig


class BurgersDatasetConfig(ConfigBase):
    folder: str = "neuralop/data/datasets/data/"
    batch_size: int = 16
    n_train: int = 800
    test_batch_sizes: List[int] = [16]
    n_tests: List[int] = [400]
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
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    eval_interval: int = 1
    mixed_precision: bool = False
    scheduler: str = "StepLR"
    scheduler_patience: int = 100
    step_size: int = 60
    gamma: float = 0.5


class RNO_Small1d(ModelConfig):
    model_arch: str = "rno"
    data_channels: int = 1
    out_channels: int = 1
    # 1D spatial, use small number of modes
    n_modes: List[int] = [8]
    hidden_channels: int = 32
    n_layers: int = 2
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    residual: bool = False
    use_channel_mlp: bool = True
    channel_mlp_dropout: float = 0.0
    channel_mlp_expansion: float = 0.5
    fno_skip: str = "linear"
    norm: Optional[str] = None
    resolution_scaling_factor: Optional[Any] = None
    domain_padding: Optional[Any] = None
    fno_block_precision: str = "full"
    stabilizer: Optional[str] = None
    max_n_modes: Optional[Any] = None
    factorization: Optional[str] = None
    rank: float = 1.0
    fixed_rank_modes: bool = False
    implementation: str = "factorized"
    separable: bool = False
    preactivation: bool = False


class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "rno"
    distributed: DistributedConfig = DistributedConfig()
    model: RNO_Small1d = RNO_Small1d()
    opt: BurgersOptConfig = BurgersOptConfig()
    data: BurgersDatasetConfig = BurgersDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()


