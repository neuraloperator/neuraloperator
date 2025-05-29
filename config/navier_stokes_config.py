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
    n_modes: List[int] = [64, 64]
    hidden_channels: int = 64
    projection_channel_ratio: int = 4
    n_layers: int = 4
    domain_padding: float = 0.0
    domain_padding_mode: str = "one-sided"
    fft_norm: str = "forward"
    norm: Optional[Any] = None
    skip: str = "linear"
    implementation: str = "reconstructed"
    use_channel_mlp: int = 1
    channel_mlp_expansion: float = 0.5
    channel_mlp_dropout: int = 0
    separable: bool = False
    factorization: Optional[Any] = None
    rank: float = 1.0
    fixed_rank_modes: Optional[Any] = None
    dropout: float = 0.0
    tensor_lasso_penalty: float = 0.0
    joint_factorization: bool = False
    stabilizer: Optional[Any] = None


class Opt(ConfigBase):
    n_epochs: int = 500
    learning_rate: float = 3e-4
    training_loss: str = "h1"
    weight_decay: float = 1e-4
    amp_autocast: bool = False
    scheduler_T_max: int = 500
    scheduler_patience: int = 50
    scheduler: str = "StepLR"
    step_size: int = 100
    gamma: float = 0.5


class Data(ConfigBase):
    folder: str = "data/navier_stokes/"
    batch_size: int = 8
    n_train: int = 10000
    train_resolution: int = 128
    n_tests: List[int] = [2000]
    test_resolutions: List[int] = [128]
    test_batch_sizes: List[int] = [8]
    encode_input: bool = True
    encode_output: bool = True


class Patching(ConfigBase):
    levels: int = 0
    padding: int = 0
    stitching: bool = False


class Wandb(ConfigBase):
    log: bool = False
    name: Optional[Any] = None
    group: str = ""
    project: str = "train_ns"
    entity: str = "dhpitt"
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