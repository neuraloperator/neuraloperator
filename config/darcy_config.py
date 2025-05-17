from typing import List, Optional, Any 


from zencfg import ConfigBase





class Distributed(ConfigBase):
    use_distributed: bool = False
    wireup_info: str = "mpi"
    wireup_store: str = "tcp"
    model_parallel_size: int = 2
    seed: int = 666


class Tfno2d(ConfigBase):
    data_channels: int = 1
    n_modes_height: int = 16
    n_modes_width: int = 16
    hidden_channels: int = 32
    projection_channel_ratio: int = 2
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
    stabilizer: Optional[Any] = None


class Opt(ConfigBase):
    n_epochs: int = 300
    learning_rate: float = 5e-3
    training_loss: str = "h1"
    weight_decay: float = 1e-4
    amp_autocast: bool = False
    scheduler_T_max: int = 500
    scheduler_patience: int = 5
    scheduler: str = "StepLR"
    step_size: int = 60
    gamma: float = 0.5


class Data(ConfigBase):
    batch_size: int = 16
    n_train: int = 1000
    train_resolution: int = 16
    n_tests: List[int] = [100, 50]
    test_resolutions: List[int] = [16, 32]
    test_batch_sizes: List[int] = [16, 16]
    encode_input: bool = True
    encode_output: bool = False


class Patching(ConfigBase):
    levels: int = 0
    padding: int = 0
    stitching: bool = False


class Wandb(ConfigBase):
    log: bool = False
    name: Optional[Any] = None
    group: str = ""
    project: str = ""
    entity: str = ""
    sweep: bool = False
    log_output: bool = True
    eval_interval: int = 1


class DarcyConfig(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "tfno2d"
    distributed: Distributed = Distributed()
    tfno2d: Tfno2d = Tfno2d()
    opt: Opt = Opt()
    data: Data = Data()
    patching: Patching = Patching()
    wandb: Wandb = Wandb()