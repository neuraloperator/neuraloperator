from typing import Any, List, Optional


from zencfg import ConfigBase





class Distributed(ConfigBase):
    use_distributed: bool = False
    wireup_info: str = "mpi"
    wireup_store: str = "tcp"
    model_parallel_size: int = 2
    seed: int = 666


class Fno(ConfigBase):
    data_channels: int = 1
    out_channels: int = 1
    n_modes: List[int] = [8, 8]
    hidden_channels: int = 24
    lifting_channel_ratio: int = 1
    projection_channel_ratio: int = 1
    n_layers: int = 5
    domain_padding: Optional[Any] = None
    domain_padding_mode: str = "one-sided"
    fft_norm: str = "forward"
    norm: str = "group_norm"
    skip: str = "linear"
    implementation: str = "factorized"
    separable: int = 0
    preactivation: int = 0
    stabilizer: Optional[Any] = None
    use_channel_mlp: int = 1
    channel_mlp_expansion: float = 0.5
    channel_mlp_dropout: int = 0
    factorization: Optional[Any] = None
    rank: float = 0.05
    fixed_rank_modes: Optional[Any] = None
    dropout: float = 0.0
    tensor_lasso_penalty: float = 0.0
    joint_factorization: bool = False


class Opt(ConfigBase):
    n_epochs: int = 10000
    learning_rate: float = 0.0001
    training_loss: str = "l2"
    weight_decay: float = 1e-4
    amp_autocast: bool = False
    scheduler_T_max: int = 500
    scheduler_patience: int = 100
    scheduler: str = "ReduceLROnPlateau"
    step_size: int = 60
    gamma: float = 0.5
    precision_schedule: Optional[Any] = None


class Data(ConfigBase):
    folder: str = "neuralop/data/datasets/data/"
    batch_size: int = 16
    n_train: int = 800
    test_batch_sizes: List[int] = [16]
    n_tests: List[int] = [400]
    spatial_length: int = 128
    temporal_length: int = 101
    encode_input: bool = False
    encode_output: bool = False
    include_endpoint: List[bool] = [True, False]


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


class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "fno"
    distributed: Distributed = Distributed()
    fno: Fno = Fno()
    opt: Opt = Opt()
    data: Data = Data()
    patching: Patching = Patching()
    wandb: Wandb = Wandb()