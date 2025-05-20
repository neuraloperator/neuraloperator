from typing import Any, Dict, Optional


from zencfg import ConfigBase





class Distributed(ConfigBase):
    use_distributed: bool = False
    wireup_info: str = "mpi"
    wireup_store: str = "tcp"
    model_parallel_size: int = 2
    seed: int = 666


class Tfno2d(ConfigBase):
    data_channels: int = 1
    n_modes_height: int = 64
    n_modes_width: int = 64
    hidden_channels: int = 128
    projection_channels: int = 256
    n_layers: int = 4
    domain_padding: Optional[Any] = None
    domain_padding_mode: str = "one-sided"
    fft_norm: str = "forward"
    norm: str = "group_norm"
    skip: str = "linear"
    implementation: str = "factorized"
    separable: int = 0
    preactivation: int = 0
    use_channel_mlp: int = 1
    channel_mlp_expansion: float = 0.5
    channel_mlp_dropout: int = 0
    factorization: Optional[Any] = None
    rank: float = 1.0
    fixed_rank_modes: Optional[Any] = None
    dropout: float = 0.0
    tensor_lasso_penalty: float = 0.0
    joint_factorization: bool = False
    fno_block_precision: str = "full"
    stabilizer: Optional[Any] = None


class Opt(ConfigBase):
    alpha: float = 0.9
    delta: float = 0.95
    solution: Dict[str, Any] = {'n_epochs': 300, 'resume': False, 'learning_rate': '5e-3', 'training_loss': 'h1', 'weight_decay': '1e-4', 'mixed_precision': False, 'scheduler_T_max': 500, 'scheduler_patience': 5, 'scheduler': 'StepLR', 'step_size': 60, 'gamma': 0.5}
    residual: Dict[str, Any] = {'n_epochs': 300, 'learning_rate': '5e-3', 'training_loss': 'h1', 'weight_decay': '1e-4', 'mixed_precision': False, 'scheduler_T_max': 500, 'scheduler_patience': 5, 'scheduler': 'StepLR', 'step_size': 60, 'gamma': 0.5}


class Solution(ConfigBase):
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


class Residual(ConfigBase):
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


class Data(ConfigBase):
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


class Wandb(ConfigBase):
    log: bool = True
    name: str = "train-uqno"
    group: str = ""
    project: str = "uqno-darcy"
    entity: str = "YOUR_NAME"
    sweep: bool = False
    log_output: bool = True
    eval_interval: int = 1


class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "tfno2d"
    distributed: Distributed = Distributed()
    tfno2d: Tfno2d = Tfno2d()
    opt: Opt = Opt()
    data: Data = Data()
    wandb: Wandb = Wandb()