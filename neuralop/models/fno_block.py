from torch import nn
import torch.nn.functional as F

from .spectral_convolution import FactorizedSpectralConv
from .skip_connections import skip_connection
from .mlp import MLP


class FNOBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes,
                 res_scaling=None,
                 n_layers=1,
                 incremental_n_modes=None,
                 use_mlp=False, mlp=None,
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
                 fft_norm='forward',
                 **kwargs):
        super().__init__()
        self.n_dim = len(n_modes)
        self.n_modes = n_modes
        self._incremental_n_modes = incremental_n_modes
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

        self.convs = FactorizedSpectralConv(
                self.in_channels, self.out_channels, self.n_modes, 
                res_scaling=res_scaling,
                incremental_n_modes=incremental_n_modes,
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

        self.fno_skips = nn.ModuleList([skip_connection(self.in_channels, self.out_channels, type=fno_skip, n_dim=self.n_dim) for _ in range(n_layers)])

        if use_mlp:
            self.mlp = nn.ModuleList(
                [MLP(in_channels=self.out_channels, 
                     hidden_channels=int(round(self.out_channels*mlp['expansion'])),
                     dropout=mlp['dropout'], n_dim=self.n_dim) for _ in range(n_layers)]
            )
            self.mlp_skips = nn.ModuleList([skip_connection(self.out_channels, self.out_channels, type=mlp_skip, n_dim=self.n_dim) for _ in range(n_layers)])
        else:
            self.mlp = None

        if norm is None:
            self.norm = None
        elif norm == 'instance_norm':
            self.norm = nn.ModuleList([getattr(nn, f'InstanceNorm{self.n_dim}d')(num_features=self.out_channels) for _ in range(n_layers)])
        elif norm == 'group_norm':
            self.norm = nn.ModuleList([nn.GroupNorm(num_groups=1, num_channels=self.out_channels) for _ in range(n_layers)])
        elif norm == 'layer_norm':
            self.norm = nn.ModuleList([nn.LayerNorm() for _ in range(n_layers)])
        else:
            raise ValueError(f'Got {norm=} but expected None or one of [instance_norm, group_norm, layer_norm]')

    def forward(self, x, index=0):
    
        if self.preactivation:
            x = self.non_linearity(x)

            if self.norm is not None:
                x = self.norm[index](x)
    
        x_skip = self.fno_skips[index](x)

        x_fno = self.convs(x, index)

        if not self.preactivation and self.norm is not None:
            x_fno = self.norm[index](x_fno)

        x = x_fno + x_skip

        if not self.preactivation and index < (self.n_layers - index):
            x = self.non_linearity(x)

        if self.mlp is not None:
            x_skip = self.mlp_skips[index](x)

            if self.preactivation:
                if index < (self.n_layers - 1):
                    x = self.non_linearity(x)

            x = self.mlp[index](x) + x_skip

            if not self.preactivation:
                if index < (self.n_layers - 1):
                    x = self.non_linearity(x)

        return x

    def get_block(self, indices):
        """Returns a sub-FNO Block layer from the jointly parametrized main block

        The parametrization of an FNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError('A single layer is parametrized, directly use the main class.')
        
        return SubModule(self, indices)
    
    def __getitem__(self, indices):
        return self.get_block(indices)


class SubModule(nn.Module):
    """Class representing one of the sub_module from the mother joint module

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules, they all point to the same data, 
    which is shared.
    """
    def __init__(self, main_module, indices):
        super().__init__()
        self.main_module = main_module
        self.indices = indices
    
    def forward(self, x):
        return self.main_module.forward(x, self.indices)

