from typing import Any, List, Optional


from zencfg import ConfigBase





class Distributed(ConfigBase):
    use_distributed: bool = False
    wireup_info: str = "mpi"
    wireup_store: str = "tcp"
    model_parallel_size: int = 2
    seed: int = 666
    device: str = "cuda:0"


class Data(ConfigBase):
    root: str = "/home/YOURNAME/data/car-pressure-data/"
    sdf_query_resolution: int = 32
    n_train: int = 500
    n_test: int = 111
    download: bool = True


class Gino(ConfigBase):
    data_channels: int = 0
    out_channels: int = 1
    latent_feature_channels: int = 1
    gno_coord_dim: int = 3
    gno_coord_embed_dim: int = 16
    gno_radius: float = 0.033
    in_gno_transform_type: str = "linear"
    out_gno_transform_type: str = "linear"
    gno_pos_embed_type: str = "nerf"
    fno_n_modes: List[int] = [16, 16, 16]
    fno_hidden_channels: int = 64
    fno_use_channel_mlp: bool = True
    fno_norm: str = "instance_norm"
    fno_ada_in_features: int = 32
    fno_factorization: str = "tucker"
    fno_rank: float = 0.4
    fno_domain_padding: float = 0.125
    fno_channel_mlp_expansion: float = 1.0
    fno_resolution_scaling_factor: int = 1


class Opt(ConfigBase):
    n_epochs: int = 301
    learning_rate: float = 1e-3
    training_loss: str = "l2"
    testing_loss: str = "l2"
    weight_decay: float = 1e-4
    amp_autocast: bool = False
    scheduler_T_max: int = 500
    scheduler_patience: int = 5
    scheduler: str = "StepLR"
    step_size: int = 50
    gamma: float = 0.5


class Wandb(ConfigBase):
    log: bool = False
    name: Optional[Any] = None
    group: str = "drag"
    project: str = ""
    entity: str = ""
    sweep: bool = False
    log_output: bool = True
    eval_interval: int = 1


class Cfd(ConfigBase):
    arch: str = "gino"
    sample_max: int = 5000
    distributed: Distributed = Distributed()
    data: Data = Data()
    gino: Gino = Gino()
    opt: Opt = Opt()
    wandb: Wandb = Wandb()