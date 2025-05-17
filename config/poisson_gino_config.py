from typing import Any, Dict, List, Optional


from zencfg import ConfigBase





class Distributed(ConfigBase):
    use_distributed: bool = False
    wireup_info: str = "mpi"
    wireup_store: str = "tcp"
    model_parallel_size: int = 2
    seed: int = 666


class Data(ConfigBase):
    batch_size: int = 1
    test_batch_size: int = 1
    file: str = "/home/YOURNAME/data/nonlin_poisson/nonlinear_poisson.obj"
    n_train: int = 7000
    n_test: int = 3000
    n_in: int = 5000
    n_out: int = 100
    n_eval: int = 6000
    n_bound: int = 4000
    query_resolution: int = 64
    train_out_res: int = 400
    padding: int = 1
    single_instance: bool = False
    input_min: int = 100
    input_max: int = 5000
    sample_random_in: Optional[Any] = None
    sample_random_out: Optional[Any] = None
    return_queries_dict: bool = True


class Patching(ConfigBase):
    levels: int = 0
    padding: int = 0
    stitching: bool = False


class Gino(ConfigBase):
    data_channels: int = 3
    out_channels: int = 1
    projection_channel_ratio: int = 4
    gno_coord_dim: int = 2
    in_gno_pos_embed_type: Optional[Any] = None
    out_gno_pos_embed_type: str = "transformer"
    gno_embed_channels: int = 16
    gno_embed_max_positions: int = 600
    gno_use_torch_scatter: bool = True
    in_gno_radius: float = 0.16
    out_gno_radius: float = 0.175
    in_gno_transform_type: str = "linear"
    out_gno_transform_type: str = "linear"
    gno_reduction: str = "mean"
    gno_weighting_function: str = "half_cos"
    gno_weight_function_scale: float = 0.030625
    gno_use_open3d: bool = False
    in_gno_channel_mlp_hidden_layers: List[int] = [256, 512, 256]
    out_gno_channel_mlp_hidden_layers: List[int] = [512, 1024, 512]
    in_gno_tanh: Optional[Any] = None
    out_gno_tanh: Optional[Any] = None
    fno_n_modes: List[int] = [20, 20]
    fno_hidden_channels: int = 64
    fno_lifting_channel_ratio: int = 4
    fno_n_layers: int = 4
    fno_use_channel_mlp: bool = True
    fno_channel_mlp_expansion: float = 0.5
    fno_norm: str = "group_norm"
    fno_ada_in_features: int = 8
    fno_factorization: Optional[Any] = None
    fno_rank: float = 0.8
    fno_domain_padding: float = 0.0


class Opt(ConfigBase):
    n_epochs: int = 1000
    training_loss: List[str] = ["equation", "boundary"]
    loss_weights: Dict[str, Any] = {'mse': 1.0, 'interior': '1e-2', 'boundary': 1.0}
    pino_method: str = "autograd"
    weight_decay: float = 1e-6
    amp_autocast: bool = False
    learning_rate: float = 1e-4
    optimizer: str = "Adam"
    scheduler: str = "ReduceLROnPlateau"
    scheduler_T_max: int = 5000
    scheduler_patience: int = 2
    gamma: float = 0.9


class Loss_Weights(ConfigBase):
    mse: float = 1.0
    interior: float = 1e-2
    boundary: float = 1.0


class Wandb(ConfigBase):
    log: bool = True
    name: Optional[Any] = None
    group: Optional[Any] = None
    project: str = ""
    entity: str = ""
    sweep: bool = False
    log_output: bool = True
    log_test_interval: int = 1


class Default(ConfigBase):
    arch: str = "gino"
    verbose: bool = True
    n_params_baseline: Optional[Any] = None
    distributed: Distributed = Distributed()
    data: Data = Data()
    patching: Patching = Patching()
    gino: Gino = Gino()
    opt: Opt = Opt()
    wandb: Wandb = Wandb()