from typing import Any, List, Optional

from zencfg import ConfigBase
from config.distributed import DistributedConfig
from config.models import FNOConfig, ModelConfig, FNO_Small2d
from config.opt import OptimizationConfig, PatchingConfig
from config.wandb import WandbConfig


class AirfransOptConfig(OptimizationConfig):
    n_epochs: int = 500  # check cli arguments
    learning_rate: float = 1e-3
    training_loss: str = "weighted_l2"
    weight_decay: float = 1e-4
    scheduler: str = "CosineAnnealingLR"
    scheduler_T_max: int = 500 # Add this line
    mixed_precision: bool = True  
    save_interval: int = 20       
    step_size: int = 60
    gamma: float = 0.5


class AirfransDatasetConfig(ConfigBase):
    data_dir: str = "/home/timm/Projects/PIML/neuraloperator/tims/airfrans/consolidated_Cp_data"
    dataset_name: str = "airfoil_cp"
    batch_size: int = 16
    train_split: str = "aoa_train"
    train_resolution: int = 128
    test_splits: List[str] = ["aoa_test", "aoa_test"]
    test_resolutions: List[int] = [128, 256]
    test_batch_sizes: List[int] = [64, 64]
    encode_input: bool = True
    encode_output: bool = True
    xlim: float = 2.0
    ylim: float = 2.0
    encoding: str = "channel-wise"
    channel_dim: int = 1
    weights: List[float] = [2.0, 1.0, 0.0]  # Weights for [u, v, Cp] in loss calculation


class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "fno"
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNOConfig(
        data_channels=4,    # [u_input, v_input, mask_binary, sdf_fixed] - FNO will add x,y coords automatically  
        out_channels=3,     # [u_target, v_target, p_target, nut_fixed]
        n_modes=[12, 12], 
        hidden_channels=128,
        lifting_channel_ratio=4,
        projection_channel_ratio=4,
        n_layers=4,
        use_channel_mlp=True,
        channel_mlp_expansion=0.5,
        stabilizer="None"
    )
    device: str = "cuda:1"
    opt: OptimizationConfig = AirfransOptConfig()
    data: AirfransDatasetConfig = AirfransDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    save_interval: int = 50
    wandb: WandbConfig = WandbConfig()

# class FNOConfig(ModelConfig):
#     model_arch: str = "fno"
#     data_channels: int
#     out_channels: int
#     n_modes: List[int]
#     hidden_channels: int
#     lifting_channel_ratio: int = 2
#     projection_channel_ratio: int = 4
#     n_layers: int = 4
#     domain_padding: float = 0.0
#     norm: str = "None"
#     fno_skip: str = "linear"
#     implementation: str = "factorized"
#     use_channel_mlp: bool = True
#     channel_mlp_expansion: float = 0.5
#     channel_mlp_dropout: float = 0
#     separable: bool = False
#     factorization: str = "None"
#     rank: float = 1.0
#     fixed_rank_modes: bool = False
#     stabilizer: str = "None"
