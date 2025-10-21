from typing import Any, List, Optional

from zencfg import ConfigBase


class Distributed(ConfigBase):
    use_distributed: bool = True
    wireup_info: str = "mpi"
    wireup_store: str = "tcp"
    model_parallel_size: int = 2
    seed: int = 666


class FNO(ConfigBase):
    n_modes: List[int] = [64, 64]  # Default to 2D: [height, width]
    width: int = 64
    hidden_channels: int = 256
    n_layers: int = 4
    domain_padding: float = 0.078125
    norm: Optional[Any] = None
    fno_skip: str = "linear"
    use_channel_mlp: int = 0
    channel_mlp: Optional[Any] = None
    channel_mlp_expansion: float = 0.5
    channel_mlp_dropout: int = 0
    separable: bool = False
    factorization: Optional[Any] = None
    rank: float = 1.0
    fixed_rank_modes: Optional[Any] = None


class TFNO(ConfigBase):
    n_modes: List[int] = [64, 64]
    width: int = 64
    hidden_channels: int = 256
    n_layers: int = 4
    domain_padding: float = 0.078125
    norm: Optional[Any] = None
    fno_skip: str = "linear"
    use_channel_mlp: int = 0
    channel_mlp: Optional[Any] = None
    channel_mlp_expansion: float = 0.5
    channel_mlp_dropout: int = 0
    separable: bool = False
    factorization: str = "Tucker"
    rank: float = 0.1
    fixed_rank_modes: Optional[Any] = None


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


class Data(ConfigBase):
    folder: str = "/data/navier_stokes/"
    batch_size: int = 16
    n_train: int = 10000
    train_resolution: int = 128
    n_tests: List[int] = [2000, 1000]
    test_resolutions: List[int] = [128, 1024]
    test_batch_sizes: List[int] = [16, 4]
    encode_input: bool = True
    encode_output: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False


class Patching(ConfigBase):
    levels: int = 1
    padding: int = 16
    stitching: bool = True


class Wandb(ConfigBase):
    log: bool = False
    name: Optional[Any] = None
    group: str = "wandb_group"


class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "fno"
    distributed: Distributed = Distributed()
    fno: FNO = FNO()
    opt: Opt = Opt()
    data: Data = Data()
    patching: Patching = Patching()
    wandb: Wandb = Wandb()


class Original_Fno(ConfigBase):
    arch: str = "tfno"
    tfno: TFNO = TFNO()
    wandb: Wandb = Wandb()


class Distributed_Mg_Tucker(ConfigBase):
    tfno: TFNO = TFNO()
    distributed: Distributed = Distributed()
    patching: Patching = Patching()
