import torch.nn as nn
import torch.nn.functional as F
from functools import partialmethod
import torch
from .spectral_convolution import FactorizedSpectralConv
from .padding import DomainPadding
from .fno_block import FNOBlocks, resample


class Lifting(nn.Module):
    def __init__(self, in_channels, out_channels, n_dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        Conv = getattr(nn, f'Conv{n_dim}d')
        self.fc = Conv(in_channels, out_channels, 1)

    def forward(self, x):
        return self.fc(x)


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, n_dim=2, non_linearity=F.gelu):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels 
        self.non_linearity = non_linearity
        Conv = getattr(nn, f'Conv{n_dim}d')
        self.fc1 = Conv(in_channels, hidden_channels, 1)
        self.fc2 = Conv(hidden_channels, out_channels, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        return x



class FNO(nn.Module):
    """N-Dimensional Fourier Neural Operator

    Parameters
    ----------
    n_modes : int tuple
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the TFNO is inferred from ``len(n_modes)``
    hidden_channels : int
        width of the FNO (i.e. number of channels)
    in_channels : int, optional
        Number of input channels, by default 3
    out_channels : int, optional
        Number of output channels, by default 1
    lifting_channels : int, optional
        number of hidden channels of the lifting block of the FNO, by default 256
    projection_channels : int, optional
        number of hidden channels of the projection block of the FNO, by default 256
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    incremental_n_modes : None or int tuple, default is None
        * If not None, this allows to incrementally increase the number of modes in Fourier domain 
          during training. Has to verify n <= N for (n, m) in zip(incremental_n_modes, n_modes).
        
        * If None, all the n_modes are used.

        This can be updated dynamically during training.
    use_mlp : bool, optional
        Whether to use an MLP layer after each FNO block, by default False
    mlp : dict, optional
        Parameters of the MLP, by default None
        {'expansion': float, 'dropout': float}
    non_linearity : nn.Module, optional
        Non-Linearity module to use, by default F.gelu
    norm : F.module, optional
        Normalization layer to use, by default None
    preactivation : bool, default is False
        if True, use resnet-style preactivation
    skip : {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use, by default 'soft-gating'
    separable : bool, default is False
        if True, use a depthwise separable spectral convolution
    factorization : str or None, {'tucker', 'cp', 'tt'}
        Tensor factorization of the parameters weight to use, by default None.
        * If None, a dense tensor parametrizes the Spectral convolutions
        * Otherwise, the specified tensor factorization is used.
    joint_factorization : bool, optional
        Whether all the Fourier Layers should be parametrized by a single tensor (vs one per layer), by default False
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of the decomposition
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
    domain_padding : None or float, optional
        If not None, percentage of padding to use, by default None
    domain_padding_mode : {'symmetric', 'one-sided'}, optional
        How to perform domain padding, by default 'one-sided'
    fft_norm : str, optional
        by default 'forward'
    """
    def __init__(self, n_modes, hidden_channels,
                 in_channels=3, 
                 out_channels=1,
                 lifting_channels=256,
                 projection_channels=256,
                 n_layers=4,
                 output_scaling_factor=None,
                 incremental_n_modes=None,
                 use_mlp=False, mlp_dropout=0, mlp_expansion=0.5,
                 non_linearity=F.gelu,
                 norm=None, preactivation=False,
                 fno_skip='linear',
                 mlp_skip='soft-gating',
                 separable=False,
                 factorization=None,
                 rank=1.0,
                 joint_factorization=False, 
                 fixed_rank_modes=False,
                 implementation='factorized',
                 decomposition_kwargs=dict(),
                 domain_padding=None,
                 domain_padding_mode='one-sided',
                 fft_norm='forward',
                 SpectralConv=FactorizedSpectralConv,
                 **kwargs):
        super().__init__()
        self.n_dim = len(n_modes)
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = fno_skip,
        self.mlp_skip = mlp_skip,
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation

        # See the class' property for underlying mechanism
        # When updated, change should be reflected in fno blocks
        self._incremental_n_modes = incremental_n_modes

        if domain_padding is not None and domain_padding > 0:
            self.domain_padding = DomainPadding(domain_padding=domain_padding, padding_mode=domain_padding_mode)
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode

        if output_scaling_factor is not None and not joint_factorization:
            if isinstance(output_scaling_factor, (float, int)):
                output_scaling_factor = [output_scaling_factor]*self.n_layers
        self.output_scaling_factor = output_scaling_factor

        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels, 
            n_modes=self.n_modes,
            output_scaling_factor=output_scaling_factor,
            use_mlp=use_mlp, mlp_dropout=mlp_dropout, mlp_expansion=mlp_expansion,
            non_linearity=non_linearity,
            norm=norm, preactivation=preactivation,
            fno_skip=fno_skip,
            mlp_skip=mlp_skip,
            incremental_n_modes=incremental_n_modes,
            rank=rank,
            fft_norm=fft_norm,
            fixed_rank_modes=fixed_rank_modes, 
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            SpectralConv=SpectralConv,
            n_layers=n_layers)

        self.lifting = Lifting(in_channels=in_channels, out_channels=self.hidden_channels, n_dim=self.n_dim)
        self.projection = Projection(in_channels=self.hidden_channels, out_channels=out_channels, hidden_channels=projection_channels,
                                     non_linearity=non_linearity, n_dim=self.n_dim)

    def forward(self, x):
        """TFNO's forward pass
        """
        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx)

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x

    @property
    def incremental_n_modes(self):
        return self._incremental_n_modes

    @incremental_n_modes.setter
    def incremental_n_modes(self, incremental_n_modes):
        self.fno_blocks.incremental_n_modes = incremental_n_modes


class FNO1d(FNO):
    """1D Fourier Neural Operator

    Parameters
    ----------
    modes_height : int
        number of Fourier modes to keep along the height
    hidden_channels : int
        width of the FNO (i.e. number of channels)
    in_channels : int, optional
        Number of input channels, by default 3
    out_channels : int, optional
        Number of output channels, by default 1
    lifting_channels : int, optional
        number of hidden channels of the lifting block of the FNO, by default 256
    projection_channels : int, optional
        number of hidden channels of the projection block of the FNO, by default 256
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    incremental_n_modes : None or int tuple, default is None
        * If not None, this allows to incrementally increase the number of modes in Fourier domain 
          during training. Has to verify n <= N for (n, m) in zip(incremental_n_modes, n_modes).
        
        * If None, all the n_modes are used.

        This can be updated dynamically during training.
    use_mlp : bool, optional
        Whether to use an MLP layer after each FNO block, by default False
    mlp : dict, optional
        Parameters of the MLP, by default None
        {'expansion': float, 'dropout': float}
    non_linearity : nn.Module, optional
        Non-Linearity module to use, by default F.gelu
    norm : F.module, optional
        Normalization layer to use, by default None
    preactivation : bool, default is False
        if True, use resnet-style preactivation
    skip : {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use, by default 'soft-gating'
    separable : bool, default is False
        if True, use a depthwise separable spectral convolution
    factorization : str or None, {'tucker', 'cp', 'tt'}
        Tensor factorization of the parameters weight to use, by default None.
        * If None, a dense tensor parametrizes the Spectral convolutions
        * Otherwise, the specified tensor factorization is used.
    joint_factorization : bool, optional
        Whether all the Fourier Layers should be parametrized by a single tensor (vs one per layer), by default False
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of the decomposition
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
    domain_padding : None or float, optional
        If not None, percentage of padding to use, by default None
    domain_padding_mode : {'symmetric', 'one-sided'}, optional
        How to perform domain padding, by default 'one-sided'
    fft_norm : str, optional
        by default 'forward'
    """
    def __init__(
        self,
        n_modes_height,
        hidden_channels,
        in_channels=3, 
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        incremental_n_modes=None,
        n_layers=4,
        non_linearity=F.gelu,
        use_mlp=False, mlp_dropout=0, mlp_expansion=0.5,
        norm=None,
        skip='soft-gating',
        separable=False,
        preactivation=False,
        factorization=None, 
        rank=1.0,
        joint_factorization=False, 
        fixed_rank_modes=False,
        implementation='factorized',
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode='one-sided',
        fft_norm='forward',
        **kwargs):
        super().__init__(
            n_modes=(n_modes_height, ),
            hidden_channels=hidden_channels,
            in_channels=in_channels, 
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            non_linearity=non_linearity,
            use_mlp=use_mlp, mlp_dropout=mlp_dropout, mlp_expansion=mlp_expansion,
            incremental_n_modes=incremental_n_modes,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization, 
            rank=rank,
            joint_factorization=joint_factorization, 
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            fft_norm=fft_norm
            )
        self.n_modes_height = n_modes_height


class FNO2d(FNO):
    """2D Fourier Neural Operator

    Parameters
    ----------
    n_modes_width : int
        number of modes to keep in Fourier Layer, along the width
    n_modes_height : int
        number of Fourier modes to keep along the height
    hidden_channels : int
        width of the FNO (i.e. number of channels)
    in_channels : int, optional
        Number of input channels, by default 3
    out_channels : int, optional
        Number of output channels, by default 1
    lifting_channels : int, optional
        number of hidden channels of the lifting block of the FNO, by default 256
    projection_channels : int, optional
        number of hidden channels of the projection block of the FNO, by default 256
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    incremental_n_modes : None or int tuple, default is None
        * If not None, this allows to incrementally increase the number of modes in Fourier domain 
          during training. Has to verify n <= N for (n, m) in zip(incremental_n_modes, n_modes).
        
        * If None, all the n_modes are used.

        This can be updated dynamically during training.
    use_mlp : bool, optional
        Whether to use an MLP layer after each FNO block, by default False
    mlp : dict, optional
        Parameters of the MLP, by default None
        {'expansion': float, 'dropout': float}
    non_linearity : nn.Module, optional
        Non-Linearity module to use, by default F.gelu
    norm : F.module, optional
        Normalization layer to use, by default None
    preactivation : bool, default is False
        if True, use resnet-style preactivation
    skip : {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use, by default 'soft-gating'
    separable : bool, default is False
        if True, use a depthwise separable spectral convolution
    factorization : str or None, {'tucker', 'cp', 'tt'}
        Tensor factorization of the parameters weight to use, by default None.
        * If None, a dense tensor parametrizes the Spectral convolutions
        * Otherwise, the specified tensor factorization is used.
    joint_factorization : bool, optional
        Whether all the Fourier Layers should be parametrized by a single tensor (vs one per layer), by default False
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of the decomposition
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
    domain_padding : None or float, optional
        If not None, percentage of padding to use, by default None
    domain_padding_mode : {'symmetric', 'one-sided'}, optional
        How to perform domain padding, by default 'one-sided'
    fft_norm : str, optional
        by default 'forward'
    """
    def __init__(
        self,
        n_modes_height,
        n_modes_width,
        hidden_channels,
        in_channels=3, 
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        incremental_n_modes=None,
        non_linearity=F.gelu,
        use_mlp=False, mlp_dropout=0, mlp_expansion=0.5,
        norm=None,
        skip='soft-gating',
        separable=False,
        preactivation=False,
        factorization=None, 
        rank=1.0,
        joint_factorization=False, 
        fixed_rank_modes=False,
        implementation='factorized',
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode='one-sided',
        fft_norm='forward',
        **kwargs):
        super().__init__(
            n_modes=(n_modes_height, n_modes_width),
            hidden_channels=hidden_channels,
            in_channels=in_channels, 
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            non_linearity=non_linearity,
            use_mlp=use_mlp, mlp_dropout=mlp_dropout, mlp_expansion=mlp_expansion,
            incremental_n_modes=incremental_n_modes,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization, 
            rank=rank,
            joint_factorization=joint_factorization, 
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            fft_norm=fft_norm
            )
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width



class FNO3d(FNO):
    """3D Fourier Neural Operator

    Parameters
    ----------
    modes_width : int
        number of modes to keep in Fourier Layer, along the width
    modes_height : int
        number of Fourier modes to keep along the height    
    modes_depth : int
        number of Fourier modes to keep along the depth
    hidden_channels : int
        width of the FNO (i.e. number of channels)
    in_channels : int, optional
        Number of input channels, by default 3
    out_channels : int, optional
        Number of output channels, by default 1
    lifting_channels : int, optional
        number of hidden channels of the lifting block of the FNO, by default 256
    projection_channels : int, optional
        number of hidden channels of the projection block of the FNO, by default 256
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    incremental_n_modes : None or int tuple, default is None
        * If not None, this allows to incrementally increase the number of modes in Fourier domain 
          during training. Has to verify n <= N for (n, m) in zip(incremental_n_modes, n_modes).
        
        * If None, all the n_modes are used.

        This can be updated dynamically during training.
    use_mlp : bool, optional
        Whether to use an MLP layer after each FNO block, by default False
    mlp : dict, optional
        Parameters of the MLP, by default None
        {'expansion': float, 'dropout': float}
    non_linearity : nn.Module, optional
        Non-Linearity module to use, by default F.gelu
    norm : F.module, optional
        Normalization layer to use, by default None
    preactivation : bool, default is False
        if True, use resnet-style preactivation
    skip : {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use, by default 'soft-gating'
    separable : bool, default is False
        if True, use a depthwise separable spectral convolution
    factorization : str or None, {'tucker', 'cp', 'tt'}
        Tensor factorization of the parameters weight to use, by default None.
        * If None, a dense tensor parametrizes the Spectral convolutions
        * Otherwise, the specified tensor factorization is used.
    joint_factorization : bool, optional
        Whether all the Fourier Layers should be parametrized by a single tensor (vs one per layer), by default False
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of the decomposition
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
    domain_padding : None or float, optional
        If not None, percentage of padding to use, by default None
    domain_padding_mode : {'symmetric', 'one-sided'}, optional
        How to perform domain padding, by default 'one-sided'
    fft_norm : str, optional
        by default 'forward'
    """
    def __init__(self,                  
        n_modes_height,
        n_modes_width,
        n_modes_depth,
        hidden_channels,
        in_channels=3, 
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        incremental_n_modes=None,
        non_linearity=F.gelu,
        use_mlp=False, mlp_dropout=0, mlp_expansion=0.5,
        norm=None,
        skip='soft-gating',
        separable=False,
        preactivation=False,
        factorization=None, 
        rank=1.0,
        joint_factorization=False, 
        fixed_rank_modes=False,
        implementation='factorized',
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode='one-sided',
        fft_norm='forward',
        **kwargs):
        super().__init__(
            n_modes=(n_modes_height, n_modes_width, n_modes_depth),
            hidden_channels=hidden_channels,
            in_channels=in_channels, 
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            non_linearity=non_linearity,
            incremental_n_modes=incremental_n_modes,
            use_mlp=use_mlp, mlp_dropout=mlp_dropout, mlp_expansion=mlp_expansion,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization, 
            rank=rank,
            joint_factorization=joint_factorization, 
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            fft_norm=fft_norm
            )
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width
        self.n_modes_height = n_modes_height


def partialclass(new_name, cls, *args, **kwargs):
    """Create a new class with different default values

    Notes
    -----
    An obvious alternative would be to use functools.partial
    >>> new_class = partial(cls, **kwargs)

    The issue is twofold:
    1. the class doesn't have a name, so one would have to set it explicitly:
    >>> new_class.__name__ = new_name

    2. the new class will be a functools object and one cannot inherit from it.

    Instead, here, we define dynamically a new class, inheriting from the existing one.
    """
    __init__ = partialmethod(cls.__init__, *args, **kwargs)
    new_class = type(new_name, (cls,),  {
        '__init__': __init__,
        '__doc__': cls.__doc__,
        'forward': cls.forward, 
    })
    return new_class


TFNO   = partialclass('TFNO', FNO, factorization='Tucker')
TFNO1d = partialclass('TFNO1d', FNO1d, factorization='Tucker')
TFNO2d = partialclass('TFNO2d', FNO2d, factorization='Tucker')
TFNO3d = partialclass('TFNO3d', FNO3d, factorization='Tucker')
