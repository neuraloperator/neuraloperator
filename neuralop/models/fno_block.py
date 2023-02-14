from torch import nn
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
    kept_modes : int tuple
        total number of modes to keep in Fourier Layer, along each dim
    separable : bool, default is True
    scale : float or 'auto', default is 'auto'
        scale to use for the init
    n_layers : int, optional
        Number of Fourier Layers, by default 4
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
    def __init__(self, in_channels, out_channels, n_modes, n_layers=1, scale='auto', separable=False,
                 fft_norm='backward', bias=True, implementation='reconstructed', joint_factorization=False,
                 rank=0.5, factorization='cp', fixed_rank_modes=False, decomposition_kwargs=dict()):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.order = len(n_modes)

        # We index quadrands only
        # n_modes is the total number of modes kept along each dimension
        # half_modes is half of that except in the last mode, correponding to the number of modes to keep in *each* quadrant for each dim
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        half_modes = [m//2 for m in n_modes]
        self.half_modes = half_modes

        self.rank = rank
        self.factorization = factorization
        self.n_layers = n_layers
        self.implementation = implementation

        if scale == 'auto':
            scale = (1 / (in_channels * out_channels))

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes=[0]
            else:
                fixed_rank_modes=None

        self.mlp = None

        self.fft_norm = fft_norm

        # Make sure we are using a Complex Factorized Tensor
        if factorization is None:
            factorization = 'Dense' # No factorization
        if not factorization.lower().startswith('complex'):
            factorization = f'Complex{factorization}'
    
        if separable:
            if in_channels != out_channels:
                raise ValueError('To use separable Fourier Conv, in_channels must be equal to out_channels, ',
                                 f'but got {in_channels=} and {out_channels=}')
            weight_shape = (in_channels, *self.half_modes)
        else:
            weight_shape = (in_channels, out_channels, *self.half_modes)
        self.separable = separable

        if joint_factorization:
            self.weight = FactorizedTensor.new(((2**(self.order-1))*n_layers, *weight_shape),
                                                rank=self.rank, factorization=factorization, 
                                                fixed_rank_modes=fixed_rank_modes,
                                                **decomposition_kwargs)
            self.weight.normal_(0, scale)
        else:
            self.weight = nn.ModuleList([
                 FactorizedTensor.new(
                    weight_shape,
                    rank=self.rank, factorization=factorization, 
                    fixed_rank_modes=fixed_rank_modes,
                    **decomposition_kwargs
                    ) for _ in range((2**(self.order-1))*n_layers)]
                )
            for w in self.weight:
                w.normal_(0, scale)

        self._contract = get_contract_fun(self.weight[0], implementation=implementation, separable=separable)

        if bias:
            self.bias = nn.Parameter(scale * torch.randn(*((n_layers, self.out_channels) + (1, )*self.order)))
        else:
            self.bias = None

    def forward(self, x, indices=0):
        """Generic forward pass for the Factorized Spectral Conv
        
        Parameters 
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
        indices : int, default is 0
            if joint_factorization, index of the layers for n_layers > 1
        
        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, *mode_sizes = x.shape
        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1]//2 + 1 # Redundant last coefficient
        
        #Compute Fourier coeffcients 
        fft_dims = list(range(-self.order, 0))
        x = torch.fft.rfftn(x.float(), norm=self.fft_norm, dim=fft_dims)

        out_fft = torch.zeros([batchsize, self.out_channels, *fft_size], device=x.device, dtype=torch.cfloat)
        
        # We contract all corners of the Fourier coefs
        # Except for the last mode: there, we take all coefs as redundant modes were already removed
        mode_indexing = [((None, m), (-m, None)) for m in self.half_modes[:-1]] + [((None, self.half_modes[-1]), )]

        for i, boundaries in enumerate(itertools.product(*mode_indexing)):
            # Keep all modes for first 2 modes (batch-size and channels)
            idx_tuple = [slice(None), slice(None)] + [slice(*b) for b in boundaries]
            
            # For 2D: [:, :, :height, :width] and [:, :, -height:, width]
            out_fft[idx_tuple] = self._contract(x[idx_tuple], self.weight[indices + i], separable=self.separable)

        x = torch.fft.irfftn(out_fft, s=(mode_sizes), norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution

        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError('A single convolution is parametrized, directly use the main class.')
        
        return SubConv2d(self, indices)
    
    def __getitem__(self, indices):
        return self.get_conv(indices)



class SubConv2d(nn.Module):
    """Class representing one of the convolutions from the mother joint factorized convolution

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules, they all point to the same data, 
    which is shared.
    """
    def __init__(self, main_conv, indices):
        super().__init__()
        self.main_conv = main_conv
        self.indices = indices
    
    def forward(self, x):
        return self.main_conv.forward(x, self.indices)


class FactorizedSpectralConv1d(FactorizedSpectralConv):
    def __init__(
        self, in_channels, out_channels, modes_height,
        n_layers=1, scale='auto', separable=False,
        fft_norm='backward', bias=True, implementation='reconstucted',
        joint_factorization=False, rank=0.5, factorization='cp',
        fixed_rank_modes=False, decomposition_kwargs=dict()):
        super().__init__(in_channels, out_channels, (modes_height, ),
            n_layers=n_layers, scale=scale, separable=separable,
            fft_norm=fft_norm, bias=bias, implementation=implementation,
            joint_factorization=joint_factorization, rank=rank, 
            factorization=factorization, fixed_rank_modes=fixed_rank_modes,
            decomposition_kwargs=decomposition_kwargs
            )
        self.half_modes_height = self.half_modes[0]

    def forward(self, x, indices=0):
        batchsize, channels, width = x.shape

        x = torch.fft.rfft(x, norm=self.fft_norm)

        out_fft = torch.zeros([batchsize, self.out_channels,  width//2 + 1], device=x.device, dtype=torch.cfloat)
        out_fft[:, :, :self.half_modes_height] = self._contract(x[:, :, :self.half_modes_height], self.weight[indices], separable=self.separable)

        x = torch.fft.irfft(out_fft, n=width, norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x


class FactorizedSpectralConv2d(FactorizedSpectralConv):
    def __init__(self, in_channels, out_channels, modes_height, modes_width, n_layers=1, scale='auto', separable=False,
                 fft_norm='backward', bias=True, implementation='reconstucted', joint_factorization=False,
                 rank=0.5, factorization='cp', fixed_rank_modes=False, decomposition_kwargs=dict()):
        super().__init__(
            in_channels, out_channels, (modes_height, modes_width),
            n_layers=n_layers, scale=scale, separable=separable,
            fft_norm=fft_norm, bias=bias, implementation=implementation,
            joint_factorization=joint_factorization, rank=rank, 
            factorization=factorization, fixed_rank_modes=fixed_rank_modes,
            decomposition_kwargs=decomposition_kwargs
            )
        self.half_modes_height, self.half_modes_width = self.half_modes

    def forward(self, x, indices=0):
        batchsize, channels, height, width = x.shape

        x = torch.fft.rfft2(x.float(), norm=self.fft_norm)

        # The output will be of size (batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1)
        out_fft = torch.zeros([batchsize, self.out_channels, height, width//2 + 1], dtype=x.dtype, device=x.device)

        # upper block (truncate high freq)
        out_fft[:, :, :self.half_modes_height, :self.half_modes_width] = self._contract(x[:, :, :self.half_modes_height, :self.half_modes_width], 
                                                                              self.weight[2*indices], separable=self.separable)
        # Lower block
        out_fft[:, :, -self.half_modes_height:, :self.half_modes_width] = self._contract(x[:, :, -self.half_modes_height:, :self.half_modes_width],
                                                                               self.weight[2*indices + 1], separable=self.separable)

        x = torch.fft.irfft2(out_fft, s=(height, width), dim=(-2, -1), norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x


class FactorizedSpectralConv3d(FactorizedSpectralConv):
    def __init__(
        self, in_channels, out_channels, 
        modes_height, modes_width, modes_depth, 
        n_layers=1, scale='auto', separable=False,
        fft_norm='backward', bias=True, implementation='reconstucted',
         joint_factorization=False, rank=0.5, factorization='cp',
          fixed_rank_modes=False, decomposition_kwargs=dict()):
        super().__init__(in_channels, out_channels, (modes_height, modes_width, modes_depth),            
            n_layers=n_layers, scale=scale, separable=separable,
            fft_norm=fft_norm, bias=bias, implementation=implementation,
            joint_factorization=joint_factorization, rank=rank, 
            factorization=factorization, fixed_rank_modes=fixed_rank_modes,
            decomposition_kwargs=decomposition_kwargs
            )
        self.half_modes_height, self.half_modes_width, self.half_modes_depth = self.half_modes

    def forward(self, x, indices=0):
        batchsize, channels, height, width, depth = x.shape

        x = torch.fft.rfftn(x.float(), norm=self.fft_norm, dim=[-3, -2, -1])

        out_fft = torch.zeros([batchsize, self.out_channels, height, width, depth//2 + 1], device=x.device, dtype=torch.cfloat)

        out_fft[:, :, :self.half_modes_height, :self.half_modes_width, :self.half_modes_depth] = self._contract(
            x[:, :, :self.half_modes_height, :self.half_modes_width, :self.half_modes_depth], self.weight[indices + 0], separable=self.separable)
        out_fft[:, :, :self.half_modes_height, -self.half_modes_width:, :self.half_modes_depth] = self._contract(
            x[:, :, :self.half_modes_height, -self.half_modes_width:, :self.half_modes_depth], self.weight[indices + 1], separable=self.separable)
        out_fft[:, :, -self.half_modes_height:, :self.half_modes_width, :self.half_modes_depth] = self._contract(
            x[:, :, -self.half_modes_height:, :self.half_modes_width, :self.half_modes_depth], self.weight[indices + 2], separable=self.separable)
        out_fft[:, :, -self.half_modes_height:, -self.half_modes_width:, :self.half_modes_depth] = self._contract(
            x[:, :, -self.half_modes_height:, -self.half_modes_width:, :self.half_modes_depth], self.weight[indices + 3], separable=self.separable)

        x = torch.fft.irfftn(out_fft, s=(height, width, depth), norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x
