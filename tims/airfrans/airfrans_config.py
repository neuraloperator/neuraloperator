from typing import Any, List, Optional

from zencfg import ConfigBase
from config.distributed import DistributedConfig
from config.models import ModelConfig, FNO_Small2d
from config.opt import OptimizationConfig, PatchingConfig
from config.wandb import WandbConfig


class AirfransOptConfig(OptimizationConfig):
    n_epochs: int = 1000  # check cli arguments
    learning_rate: float = 5e-3
    training_loss: str = "h1"
    weight_decay: float = 1e-4
    scheduler: str = "CosineAnnealingLR"
    scheduler_T_max: int = 100 # Add this line
    mixed_precision: bool = True  
    save_interval: int = 20       
    step_size: int = 60
    gamma: float = 0.5


class AirfransDatasetConfig(ConfigBase):
    data_dir: str = "/home/timm/Projects/PIML/neuraloperator/tims/airfrans/consolidated_Cp_data"
    dataset_name: str = "airfoil_cp"
    batch_size: int = 64
    train_split: str = "scarce_train"
    train_resolution: int = 64
    test_splits: List[str] = ["full_test", "aoa_test"]
    test_resolutions: List[int] = [64, 128]
    test_batch_sizes: List[int] = [64, 64]
    encode_input: bool = True
    encode_output: bool = True
    xlim: float = 2.0
    ylim: float = 2.0
    encoding: str = "channel-wise"
    channel_dim: int = 1



class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "fno"
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNO_Small2d(
        data_channels=4,    # [u_input, v_input, mask_binary, sdf_fixed] - FNO will add x,y coords automatically  
        out_channels=3,     # [u_target, v_target, p_target, nut_fixed]
        n_modes=[16, 16], 
        hidden_channels=64
    )
    opt: OptimizationConfig = AirfransOptConfig()
    data: AirfransDatasetConfig = AirfransDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    save_interval: int = 50
    wandb: WandbConfig = WandbConfig()

