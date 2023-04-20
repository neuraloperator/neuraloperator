from torch import nn
import torch.nn.functional as F
import torch
from .spectral_convolution import FactorizedSpectralConv
from .skip_connections import skip_connection
from .resample import resample
from .mlp import MLP

class FNOBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes,
                 output_scaling_factor=None,
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
                 SpectralConv=FactorizedSpectralConv,
                 joint_factorization=False, 
                 fixed_rank_modes=False,
                 implementation='factorized',
                 decomposition_kwargs=dict(),
                 fft_norm='forward',
                 **kwargs):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        self.n_dim = len(n_modes)

        if output_scaling_factor is not None:
            if isinstance(output_scaling_factor, (float, int)):
                output_scaling_factor = [float(output_scaling_factor)]*len(self.n_modes)
        self.output_scaling_factor = output_scaling_factor

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

        self.convs = SpectralConv(
                self.in_channels, self.out_channels, self.n_modes, 
                output_scaling_factor=output_scaling_factor,
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
            self.mlp_skips = nn.ModuleList([skip_connection(self.in_channels, self.out_channels, type=mlp_skip, n_dim=self.n_dim) for _ in range(n_layers)])
        else:
            self.mlp = None

        # Each block will have 2 norms if we also use an MLP
        self.n_norms = 1 if self.mlp is None else 2
        if norm is None:
            self.norm = None
        elif norm == 'instance_norm':
            self.norm = nn.ModuleList([getattr(nn, f'InstanceNorm{self.n_dim}d')(num_features=self.out_channels) for _ in range(n_layers*self.n_norms)])
        elif norm == 'group_norm':
            self.norm = nn.ModuleList([nn.GroupNorm(num_groups=1, num_channels=self.out_channels) for _ in range(n_layers*self.n_norms)])
        elif norm == 'layer_norm':
            self.norm = nn.ModuleList([nn.LayerNorm(elementwise_affine=False) for _ in range(n_layers*self.n_norms)])
        else:
            raise ValueError(f'Got {norm=} but expected None or one of [instance_norm, group_norm, layer_norm]')

    def forward(self, x, index=0):
        
        if self.preactivation:
            x = self.non_linearity(x)

            if self.norm is not None:
                x = self.norm[self.n_norms*index](x)
    
        x_skip_fno = self.fno_skips[index](x)
        if self.convs.output_scaling_factor is not None:
            x_skip_fno = resample(x_skip_fno, self.convs.output_scaling_factor, list(range(-len(self.convs.output_scaling_factor), 0)))

        if self.mlp is not None:
            x_skip_mlp = self.mlp_skips[index](x)
            if self.convs.output_scaling_factor is not None:
                x_skip_mlp = resample(x_skip_mlp, self.convs.output_scaling_factor, list(range(-len(self.convs.output_scaling_factor), 0)))

        x_fno = self.convs(x, index)

        if not self.preactivation and self.norm is not None:
            x_fno = self.norm[self.n_norms*index](x_fno)
    
        x = x_fno + x_skip_fno

        if not self.preactivation and (self.mlp is not None) or (index < (self.n_layers - index)):
            x = self.non_linearity(x)

        if self.mlp is not None:
            # x_skip = self.mlp_skips[index](x)

            if self.preactivation:
                if index < (self.n_layers - 1):
                    x = self.non_linearity(x)

                if self.norm is not None:
                    x = self.norm[self.n_norms*index+1](x)

            x = self.mlp[index](x) + x_skip_mlp

            if not self.preactivation and self.norm is not None:
                x = self.norm[self.n_norms*index+1](x)

            if not self.preactivation:
                if index < (self.n_layers - 1):
                    x = self.non_linearity(x)
        return x

    @property
    def incremental_n_modes(self):
        return self._incremental_n_modes

    @incremental_n_modes.setter
    def incremental_n_modes(self, incremental_n_modes):
        self.convs.incremental_n_modes = incremental_n_modes

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

