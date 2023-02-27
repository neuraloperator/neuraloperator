import torch.nn as nn
import torch.nn.functional as F
from functools import partial, partialmethod

from .mlp import MLP
from .fno_block import FactorizedSpectralConv3d, FactorizedSpectralConv2d, FactorizedSpectralConv1d
from .fno_block import FactorizedSpectralConv
from .skip_connections import skip_connection
from .padding import DomainPadding


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
                 use_mlp=False, mlp=None,
                 non_linearity=F.gelu,
                 norm=None, preactivation=False,
                 skip='soft-gating',
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
        self.skip = skip,
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation

        if domain_padding is not None and domain_padding > 0:
            self.domain_padding = DomainPadding(domain_padding=domain_padding, padding_mode=domain_padding_mode)
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode

        self.convs = FactorizedSpectralConv(
            self.hidden_channels, self.hidden_channels, self.n_modes, 
            rank=rank,
            fft_norm=fft_norm,
            fixed_rank_modes=fixed_rank_modes, 
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            n_layers=n_layers,
        )

        self.fno_skips = nn.ModuleList([skip_connection(self.hidden_channels, self.hidden_channels, type=skip, n_dim=self.n_dim) for _ in range(n_layers)])

        if use_mlp:
            self.mlp = nn.ModuleList(
                [MLP(in_channels=self.hidden_channels, hidden_channels=int(round(self.hidden_channels*mlp['expansion'])),
                     dropout=mlp['dropout'], n_dim=self.n_dim) for _ in range(n_layers)]
            )
            self.mlp_skips = nn.ModuleList([skip_connection(self.hidden_channels, self.hidden_channels, type=skip, n_dim=self.n_dim) for _ in range(n_layers)])
        else:
            self.mlp = None

        if norm is None:
            self.norm = None
        elif norm == 'instance_norm':
            self.norm = nn.ModuleList([getattr(nn, f'InstanceNorm{self.n_dim}d')(num_features=self.hidden_channels) for _ in range(n_layers)])
        elif norm == 'group_norm':
            self.norm = nn.ModuleList([nn.GroupNorm(num_groups=1, num_channels=self.hidden_channels) for _ in range(n_layers)])
        elif norm == 'layer_norm':
            self.norm = nn.ModuleList([nn.LayerNorm() for _ in range(n_layers)])
        else:
            raise ValueError(f'Got {norm=} but expected None or one of [instance_norm, group_norm, layer_norm]')

        self.lifting = Lifting(in_channels=in_channels, out_channels=self.hidden_channels, n_dim=self.n_dim)
        self.projection = Projection(in_channels=self.hidden_channels, out_channels=out_channels, hidden_channels=projection_channels,
                                     non_linearity=non_linearity, n_dim=self.n_dim)


    def forward(self, x):
        """TFNO's forward pass
        """
        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for i in range(self.n_layers):

            if self.preactivation:
                x = self.non_linearity(x)

                if self.norm is not None:
                    x = self.norm[i](x)

            x_fno = self.convs[i](x)

            if not self.preactivation and self.norm is not None:
                x_fno = self.norm[i](x_fno)

            x_skip = self.fno_skips[i](x)
            x = x_fno + x_skip

            if not self.preactivation and i < (self.n_layers - 1):
                x = self.non_linearity(x)

            if self.mlp is not None:
                x_skip = self.mlp_skips[i](x)

                if self.preactivation:
                    if i < (self.n_layers - 1):
                        x = self.non_linearity(x)

                x = self.mlp[i](x) + x_skip

                if not self.preactivation:
                    if i < (self.n_layers - 1):
                        x = self.non_linearity(x)


        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)
        return x


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
        n_layers=4,
        non_linearity=F.gelu,
        use_mlp=False, mlp=None,
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
            use_mlp=use_mlp, mlp=mlp,
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

        self.convs = FactorizedSpectralConv1d(
            self.hidden_channels, self.hidden_channels, self.n_modes_height, 
            rank=rank,
            fft_norm=fft_norm,
            fixed_rank_modes=fixed_rank_modes, 
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            n_layers=n_layers,
        )


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
        non_linearity=F.gelu,
        use_mlp=False, mlp=None,
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
            use_mlp=use_mlp, mlp=mlp,
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

        self.convs = FactorizedSpectralConv2d(
            self.hidden_channels, self.hidden_channels,
            self.n_modes_height, self.n_modes_width, 
            rank=rank,
            fft_norm=fft_norm,
            fixed_rank_modes=fixed_rank_modes, 
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            n_layers=n_layers,
        )


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
        non_linearity=F.gelu,
        use_mlp=False, mlp=None,
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
            use_mlp=use_mlp, mlp=mlp,
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

        self.convs = FactorizedSpectralConv3d(
            self.hidden_channels, self.hidden_channels, 
            self.n_modes_height, self.n_modes_width, self.n_modes_height,
            rank=rank,
            fft_norm=fft_norm,
            fixed_rank_modes=fixed_rank_modes, 
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            n_layers=n_layers,
        )



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
