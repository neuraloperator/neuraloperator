from typing import List, Literal
from zencfg import ConfigBase

class FNOConfig(ConfigBase):
    data_channels: int
    out_channels: int
    n_modes: List[int]
    hidden_channels: int
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 4
    n_layers: int = 4
    domain_padding: float = 0.0
    domain_padding_mode: str = "one-sided"
    fft_norm: str = "forward"
    norm: str = "None"
    skip: str = "linear"
    implementation: str = "reconstructed"
    use_channel_mlp: int = 1
    channel_mlp_expansion: float = 0.5
    channel_mlp_dropout: float = 0
    separable: bool = False
    factorization: str = "None"
    rank: float = 1.0
    fixed_rank_modes: str = "None"
    dropout: float = 0.0
    joint_factorization: bool = False
    stabilizer: str = "None"

