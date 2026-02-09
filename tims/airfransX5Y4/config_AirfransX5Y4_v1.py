from typing import Any, List, Optional

from zencfg import ConfigBase
from config.distributed import DistributedConfig
from config.models import FNOConfig, ModelConfig, FNO_Small2d
from config.opt import OptimizationConfig, PatchingConfig
from config.wandb import WandbConfig


class AirfransOptConfig(OptimizationConfig):
    n_epochs: int = 1000  # check cli arguments
    learning_rate: float = 1e-3
    training_loss: str = "weighted_l2"
    weight_decay: float = 1e-4
    scheduler: str = "CosineAnnealingLR"
    scheduler_T_max: int = 1000 
    mixed_precision: bool = True  
    save_interval: int = 20       
    step_size: int = 60
    gamma: float = 0.5
    eval_interval: int = 20


class AirfransDatasetConfig(ConfigBase):
    data_dir: str = "/home/timm/Projects/PIML/Dataset_PT_FNO_X5Y4/TrainingX5Y4_consolidated"
    dataset_name: str = "airfoil"
    batch_size: int = 16
    train_split: str = "full_train"
    train_resolution: int = 128
    test_splits: List[str] = ["full_test", "full_test"]
    test_resolutions: List[int] = [128, 1024]
    test_batch_sizes: List[int] = [64, 64]
    encode_input: bool = True
    encode_output: bool = True
    xlim: float = 6.0
    ylim: float = 3.0
    encoding: str = "channel-wise"
    channel_dim: int = 1
    weights: List[float] = [0.6, 1.0, 0.8, 0.1]  # Weights for [u, v, Cp, log_nut_ratio] in loss calculation


class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "fno"
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNOConfig(
        data_channels=5,    # [u_input, v_input, mask_binary, sdf_fixed, logRn] - FNO will add x,y coords automatically  
        out_channels=4,     # [u_target, v_target, p_target, log_nutratio]
        n_modes=[24, 24], 
        hidden_channels=64,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        n_layers=4,
        use_channel_mlp=True,
        channel_mlp_expansion=0.5,
        stabilizer="None"
    )
    device: str = "cuda"
    opt: OptimizationConfig = AirfransOptConfig()
    data: AirfransDatasetConfig = AirfransDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    save_interval: int = 50
    wandb: WandbConfig = WandbConfig()

