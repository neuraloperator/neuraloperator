from torch import nn
import torch.nn.functional as F
import torch
import itertools

import tensorly as tl
from tensorly.plugins import use_opt_einsum
tl.set_backend('pytorch')

use_opt_einsum('optimal')

from tltorch.factorized_tensors.core import FactorizedTensor

einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

def _contract_dense(x, weight, separable=False):
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:]) # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order]) # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0] 

    eq= ''.join(x_syms) + ',' + ''.join(weight_syms) + '->' + ''.join(out_syms)

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()
    if x.shape[-1] < weight.shape[-1]:
        weight = weight[..., :x.shape[-1]]
    return tl.einsum(eq, x, weight)

def _contract_dense_separable(x, weight, separable=True):
    if separable == False:
        raise ValueError('This function is only for separable=True')
    return x*weight

def _contract_cp(x, cp_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order+1]
    out_syms = list(x_syms)
    if separable:
        factor_syms = [einsum_symbols[1]+rank_sym] #in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1]+rank_sym,out_sym+rank_sym] #in, out
    factor_syms += [xs+rank_sym for xs in x_syms[2:]] #x, y, ...
    eq = x_syms + ',' + rank_sym + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)

    return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)
 

def _contract_tucker(x, tucker_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order+1:2*order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        factor_syms = [xs+rs for (xs, rs) in zip(x_syms[1:], core_syms)] #x, y, ...

    else:
        core_syms = einsum_symbols[order+1:2*order+1]
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1]+core_syms[0], out_sym+core_syms[1]] #out, in
        factor_syms += [xs+rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])] #x, y, ...
    
    eq = x_syms + ',' + core_syms + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)

    return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def _contract_tt(x, tt_weight, separable=False):
    order = tl.ndim(x)

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:]) # no batch-size
    if not separable:
        weight_syms.insert(1, einsum_symbols[order]) # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[order+1:])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i+1]])
    eq = ''.join(x_syms) + ',' + ','.join(''.join(f) for f in tt_syms) + '->' + ''.join(out_syms)

    return tl.einsum(eq, x, *tt_weight.factors)


def get_contract_fun(weight, implementation='reconstructed', separable=False):
    """Generic ND implementation of Fourier Spectral Conv contraction
    
    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input (factorized)
    
    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == 'reconstructed':
        if separable:
            print('SEPARABLE')
            return _contract_dense_separable
        else:
            return _contract_dense
    elif implementation == 'factorized':
        if torch.is_tensor(weight):
            return _contract_dense
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower() == 'complexdense':
                return _contract_dense
            elif weight.name.lower() == 'complextucker':
                return _contract_tucker
            elif weight.name.lower() == 'complextt':
                return _contract_tt
            elif weight.name.lower() == 'complexcp':
                return _contract_cp
            else:
                raise ValueError(f'Got unexpected factorized weight type {weight.name}')
        else:
            raise ValueError(f'Got unexpected weight type of class {weight.__class__.__name__}')
    else:
        raise ValueError(f'Got {implementation=}, expected "reconstructed" or "factorized"')


class FactorizedSpectralConv(nn.Module):
    """Generic N-Dimensional Fourier Neural Operator

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels
    out_channels : int, optional
        Number of output channels
    n_modes : int tuple
        total number of modes to keep in Fourier Layer, along each dim
    separable : bool, default is True
    scale : float or 'auto', default is 'auto'
        scale to use for the init
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    incremental_n_modes : None or int tuple, default is None
        * If not None, this allows to incrementally increase the number of modes in Fourier domain 
          during training. Has to verify n <= N for (n, m) in zip(incremental_n_modes, n_modes).
        
        * If None, all the n_modes are used.

        This can be updated dynamically during training.
    joint_factorization : bool, optional
        Whether all the Fourier Layers should be parametrized by a single tensor (vs one per layer), by default False
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
    factorization : str, {'tucker', 'cp', 'tt'}, optional
        Tensor factorization of the parameters weight to use, by default 'tucker'
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    fft_norm : str, optional
        by default 'forward'
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of the decomposition
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
    """
    def __init__(self, in_channels, out_channels, n_modes, incremental_n_modes=None, bias=True,
                 n_layers=1, scale='auto', separable=False, fft_norm='backward',
                 rank=0.5, factorization='cp', implementation='reconstructed', 
                 fixed_rank_modes=False, joint_factorization=False, decomposition_kwargs=dict()):

from .spectral_convolution import FactorizedSpectralConv
from .skip_connections import skip_connection
from .resample import resample
from .mlp import MLP
from .normalization_layers import AdaIN

class FNOBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes,
                 output_scaling_factor=None,
                 n_layers=1,
                 incremental_n_modes=None,
                 use_mlp=False, mlp_dropout=0, mlp_expansion=0.5,
                 non_linearity=F.gelu,
                 norm=None, ada_in_features=None,
                 preactivation=False,
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
                output_scaling_factor = [[float(output_scaling_factor)]*len(self.n_modes)]*n_layers
            elif isinstance(output_scaling_factor[0], (float, int)):
                output_scaling_factor = [[s]*len(self.n_modes) for s in output_scaling_factor]
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
        self.fno_skip = fno_skip
        self.mlp_skip = mlp_skip
        self.use_mlp = use_mlp
        self.mlp_expansion = mlp_expansion
        self.mlp_dropout = mlp_dropout
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features

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
                     hidden_channels=int(round(self.out_channels*mlp_expansion)),
                     dropout=mlp_dropout, n_dim=self.n_dim) for _ in range(n_layers)]
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
        # elif norm == 'layer_norm':
        #     self.norm = nn.ModuleList([nn.LayerNorm(elementwise_affine=False) for _ in range(n_layers*self.n_norms)])
        elif norm == 'ada_in':
            self.norm = nn.ModuleList([AdaIN(ada_in_features, out_channels) for _ in range(n_layers*self.n_norms)])
        else:
            raise ValueError(f'Got {norm=} but expected None or one of [instance_norm, group_norm, layer_norm]')

    def set_ada_in_embeddings(self, *embeddings):
        """Sets the embeddings of each Ada-IN norm layers

        Parameters
        ----------
        embeddings : tensor or list of tensor
            if a single embedding is given, it will be used for each norm layer
            otherwise, each embedding will be used for the corresponding norm layer
        """
        if len(embeddings) == 1:
            for norm in self.norm:
                norm.set_embedding(embeddings[0])
        else:
            for norm, embedding in zip(self.norm, embeddings):
                norm.set_embedding(embedding)
        
    def forward(self, x, index=0, output_shape = None):
        
        if self.preactivation:
            x = self.non_linearity(x)

            if self.norm is not None:
                x = self.norm[self.n_norms*index](x)
    
        x_skip_fno = self.fno_skips[index](x)
        if self.convs.output_scaling_factor is not None:
            # x_skip_fno = resample(x_skip_fno, self.convs.output_scaling_factor[index], list(range(-len(self.convs.output_scaling_factor[index]), 0)))
            x_skip_fno = resample(x_skip_fno, self.output_scaling_factor[index]\
                                  , list(range(-len(self.output_scaling_factor[index]), 0)), output_shape = output_shape )


        if self.mlp is not None:
            x_skip_mlp = self.mlp_skips[index](x)
            if self.convs.output_scaling_factor is not None:
                x_skip_mlp = resample(x_skip_mlp, self.output_scaling_factor[index]\
                                      , list(range(-len(self.output_scaling_factor[index]), 0)), output_shape = output_shape )
        

        x_fno = self.convs(x, index, output_shape=output_shape)

        

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