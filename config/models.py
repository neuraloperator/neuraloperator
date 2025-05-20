from typing import List, Literal
from zencfg import ConfigBase

from neuralop.models import (FNO,
                             SFNO,
                             FNOGNO,
                             GINO)

class FNOConfig(ConfigBase):
    """FNOConfig 

    Parameters
    ----------
    n_modes : Tuple[int]
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the FNO is inferred from ``len(n_modes)``
    in_channels : int
        Number of channels in input function
    out_channels : int
        Number of channels in output function
    hidden_channels : int
        width of the FNO (i.e. number of channels)
    n_layers : int, optional
        Number of Fourier Layers, by default 4

    Documentation for more advanced parameters is below.

    Other parameters
    ------------------
    lifting_channel_ratio : int, optional
        ratio of lifting channels to hidden_channels, by default 2
        The number of liting channels in the lifting block of the FNO is
        lifting_channel_ratio * hidden_channels (e.g. default 2 * hidden_channels)
    projection_channel_ratio : int, optional
        ratio of projection channels to hidden_channels, by default 2
        The number of projection channels in the projection block of the FNO is
        projection_channel_ratio * hidden_channels (e.g. default 2 * hidden_channels)
    non_linearity : nn.Module, optional
        Non-Linear activation function module to use, by default F.gelu
    norm : Literal ["ada_in", "group_norm", "instance_norm"], optional
        Normalization layer to use, by default None
    complex_data : bool, optional
        Whether data is complex-valued (default False)
        if True, initializes complex-valued modules.
    use_channel_mlp : bool, optional
        Whether to use an MLP layer after each FNO block, by default True
    channel_mlp_dropout : float, optional
        dropout parameter for ChannelMLP in FNO Block, by default 0
    channel_mlp_expansion : float, optional
        expansion parameter for ChannelMLP in FNO Block, by default 0.5
    channel_mlp_skip : Literal['linear', 'identity', 'soft-gating'], optional
        Type of skip connection to use in channel-mixing mlp, by default 'soft-gating'
    fno_skip : Literal['linear', 'identity', 'soft-gating'], optional
        Type of skip connection to use in FNO layers, by default 'linear'
    resolution_scaling_factor : Union[Number, List[Number]], optional
        layer-wise factor by which to scale the domain resolution of function, by default None
        
        * If a single number n, scales resolution by n at each layer

        * if a list of numbers [n_0, n_1,...] scales layer i's resolution by n_i.
    domain_padding : Union[Number, List[Number]], optional
        If not None, percentage of padding to use, by default None
        To vary the percentage of padding used along each input dimension,
        pass in a list of percentages e.g. [p1, p2, ..., pN] such that
        p1 corresponds to the percentage of padding along dim 1, etc.
    domain_padding_mode : Literal ['symmetric', 'one-sided'], optional
        How to perform domain padding, by default 'symmetric'
    fno_block_precision : str {'full', 'half', 'mixed'}, optional
        precision mode in which to perform spectral convolution, by default "full"
    stabilizer : str {'tanh'} | None, optional
        whether to use a tanh stabilizer in FNO block, by default None

        Note: stabilizer greatly improves performance in the case
        `fno_block_precision='mixed'`. 

    max_n_modes : Tuple[int] | None, optional

        * If not None, this allows to incrementally increase the number of
        modes in Fourier domain during training. Has to verify n <= N
        for (n, m) in zip(max_n_modes, n_modes).

        * If None, all the n_modes are used.

        This can be updated dynamically during training.
    factorization : str, optional
        Tensor factorization of the FNO layer weights to use, by default None.

        * If None, a dense tensor parametrizes the Spectral convolutions

        * Otherwise, the specified tensor factorization is used.
    rank : float, optional
        tensor rank to use in above factorization, by default 1.0
    implementation : str {'factorized', 'reconstructed'}, optional

        * If 'factorized', implements tensor contraction with the individual factors of the decomposition 
        
        * If 'reconstructed', implements with the reconstructed full tensorized weight.
    decomposition_kwargs : dict, optional
        extra kwargs for tensor decomposition (see `tltorch.FactorizedTensor`), by default dict()
    separable : bool, optional (**DEACTIVATED**)
        if True, use a depthwise separable spectral convolution, by default False   
    preactivation : bool, optional (**DEACTIVATED**)
        whether to compute FNO forward pass with resnet-style preactivation, by default False
    conv_module : nn.Module, optional
        module to use for FNOBlock's convolutions, by default SpectralConv
    """
    model_arch: str = "fno"
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
    use_channel_mlp: bool = True
    channel_mlp_expansion: float = 0.5
    channel_mlp_dropout: float = 0
    separable: bool = False
    factorization: str = "None"
    rank: float = 1.0
    fixed_rank_modes: str = "None"
    dropout: float = 0.0
    joint_factorization: bool = False
    stabilizer: str = "None"

class SimpleFNOConfig(FNOConfig):
    """
    SimpleFNOConfig: a basic FNO config that provides access to only
    the most important FNO parameters.
    """
    data_channels: int
    out_channels: int
    n_modes: List[int]
    hidden_channels: int
    projection_channel_ratio: int

class Small2dFNO(SimpleFNOConfig):
    """
    Small2dFNO: a basic, small FNO for 2d problems.
    """
    data_channels: int = 1
    out_channels: int = 1
    n_modes: List[int] = [16,16]
    hidden_channels: int = 24
    projection_channel_ratio: int = 2

fno_param_docstring = FNO.__doc__.split("Parameters")[1:]
Small2dFNO.__doc__ += "Parameters".join(fno_param_docstring) # this doesn't really work

class FNOGNOConfig(ConfigBase):
    model_arch: str = "fnogno"
    data_channels: int
    out_channels: int
    gno_coord_dim: int
    gno_coord_embed_dim: int
    gno_radius: float
    gno_transform_type: str
    fno_n_modes: List[int]
    fno_hidden_channels: int
    fno_use_channel_mlp: bool = True
    fno_norm: str = "instance_norm"
    fno_ada_in_features: int = 32
    fno_factorization: str = "tucker"
    fno_rank: float = 1.0
    fno_domain_padding: float = 0.125
    fno_use_channel_mlp: bool = True
    fno_channel_mlp_expansion: float = 1.0
    fno_resolution_scaling_factor: int = 1

class CarCFDFNOGNOConfig(FNOGNOConfig):
    data_channels: int = 1
    out_channels: int = 1
    gno_coord_dim: int = 3
    gno_coord_embed_dim: int = 16
    gno_radius: float = 0.033
    gno_transform_type: str = "linear"
    fno_n_modes: List[int] = [16, 16, 16]
    fno_hidden_channels: int = 64
    fno_use_channel_mlp: bool = True
    fno_rank: float = 0.4
    fno_domain_padding: float = 0.125
