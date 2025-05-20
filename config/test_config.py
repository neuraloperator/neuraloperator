from typing import Any, Optional


from zencfg import ConfigBase





class Tfno2d(ConfigBase):
    data_channels: int = 3
    n_modes_height: int = 8
    n_modes_width: int = 8
    hidden_channels: int = 32
    projection_channel_ratio: int = 1
    n_layers: int = 2
    domain_padding: int = 0
    domain_padding_mode: str = "symmetric"
    fft_norm: str = "forward"
    norm: Optional[Any] = None
    skip: str = "soft-gating"
    implementation: str = "factorized"
    use_channel_mlp: int = 1
    channel_mlp_expansion: float = 0.5
    channel_mlp_dropout: int = 0
    factorization: Optional[Any] = None
    rank: float = 1.0
    fixed_rank_modes: Optional[Any] = None
    dropout: float = 0.0
    tensor_lasso_penalty: float = 0.0
    joint_factorization: bool = False
    stabilizer: Optional[Any] = None


class Data(ConfigBase):
    batch_size: int = 4
    n_train: int = 10
    size: int = 32


class Opt(ConfigBase):
    n_epochs: int = 500
    learning_rate: float = 1e-3
    training_loss: str = "h1"
    weight_decay: float = 1e-4
    mixed_precision: bool = False
    scheduler_T_max: int = 500
    scheduler_patience: int = 5
    scheduler: str = "StepLR"
    step_size: int = 100
    gamma: float = 0.5


class Patching(ConfigBase):
    levels: int = 0
    padding: int = 0
    stitching: bool = True


class TestConfig(ConfigBase):
    verbose: bool = True
    arch: str = "tfno2d"
    tfno2d: Tfno2d = Tfno2d()
    data: Data = Data()
    opt: Opt = Opt()
    patching: Patching = Patching()