from .tfno import Lifting, Projection
import torch.nn as nn
import torch.nn.functional as F
from functools import partialmethod
import torch
from .mlp import MLP
from .spectral_convolution import FactorizedSpectralConv3d, FactorizedSpectralConv2d, FactorizedSpectralConv1d
from .spectral_convolution import FactorizedSpectralConv
from .skip_connections import skip_connection
from .padding import DomainPadding
from .fno_block import FNOBlocks, resample
from .tfno import partialclass


class UNO(nn.Module):
    """N-Dimensional U-shaped Fourier Neural Operator

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 3
    out_channels : int, optional
        Number of output channels, by default 1
    hidden_channels : int
        width of the FNO (i.e. number of channels)
    lifting_channels : int, optional
        number of hidden channels of the lifting block of the FNO, by default 256
    projection_channels : int, optional
        number of hidden channels of the projection block of the FNO, by default 256
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    uno_out_channels: list
        Number of output channel of each Fourier Layers.
        Eaxmple: For a Five layer UNO uno_out_channels can be [32,64,64,64,32]
    uno_n_modes: list
        Number of Fourier Modes to use in integral operation of each Fourier Layers (along each dimension).
        Example: For a five layer UNO with 2D input the uno_n_modes can be: [[5,5],[5,5],[5,5],[5,5],[5,5]]
    uno_scalings: list
        Scaling Factors for each Fourier Layers
        Example: For a five layer UNO with 2D input, the uno_scalings can be : [[1.0,1.0],[0.5,0.5],[1,1],[1,1],[2,2]]
    horizontal_skips_map: Dict, optional
                    a map {...., b: a, ....} denoting horizontal skip connection from a-th layer to
                    b-th layer. If None default skip connection is applied.
                    Example: For a 5 layer UNO architecture, the skip connections can be 
                    horizontal_skips_map ={4:0,3:1}

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
                 in_channels, 
                 out_channels,
                 hidden_channels,
                 lifting_channels=256,
                 projection_channels=256,
                 n_layers=4,
                 uno_out_channels = None,
                 uno_n_modes = None,
                 uno_scalings = None,
                 horizontal_skips_map = None,
                 incremental_n_modes=None,
                 use_mlp=False, mlp_dropout=0, mlp_expansion=0.5,
                 non_linearity=F.gelu,
                 norm=None, preactivation=False,
                 fno_skip='linear',
                 horizontal_skip = 'linear',
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
                 **kwargs):
        super().__init__()
        self.n_layers = n_layers
        print(uno_out_channels)
        assert len(uno_out_channels) == n_layers, "Output channels for all layers are not given"
        assert len(uno_n_modes) == n_layers, "number of modes for all layers are not given"
        assert len(uno_scalings) == n_layers, "Scaling factor for all layers are not given"

        self.n_dim = len(uno_n_modes[0])
        self.uno_out_channels = uno_out_channels
        self.uno_n_modes = uno_n_modes
        self.uno_scalings = uno_scalings
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.horizontal_skips_map = horizontal_skips_map
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
        self._incremental_n_modes = incremental_n_modes
        
        if self.horizontal_skips_map is None:
            self.horizontal_skips_map = {}
            for i in range(n_layers//2,0,):
                self.horizontal_skips_map[n_layers - i -1] = i
        

        if domain_padding is not None and domain_padding > 0:
            self.domain_padding = DomainPadding(domain_padding=domain_padding, padding_mode=domain_padding_mode\
            , output_scale_factor = self.uno_scalings)
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode

        

        self.lifting = Lifting(in_channels=in_channels, out_channels=self.hidden_channels, n_dim=self.n_dim)
        self.fno_blocks = nn.ModuleList([])
        self.horizontal_skips = torch.nn.ModuleDict({})
        prev_out = self.hidden_channels
        for i in range(self.n_layers):

            if i in self.horizontal_skips_map.keys():
                prev_out = prev_out + self.uno_out_channels[self.horizontal_skips_map[i]]

            self.fno_blocks.append(FNOBlocks(
                                            in_channels=prev_out,
                                            out_channels= self.uno_out_channels[i], 
                                            n_modes=self.uno_n_modes[i],
                                            use_mlp=use_mlp, mlp_dropout=mlp_dropout, mlp_expansion=mlp_expansion,
                                            output_scaling_factor = self.uno_scalings[i],
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
                                            n_layers=1))
            
            if i in self.horizontal_skips_map.values():
                self.horizontal_skips[str(i)] = skip_connection( self.uno_out_channels[i],  \
                self.uno_out_channels[i], type=horizontal_skip, n_dim=self.n_dim)

            prev_out = self.uno_out_channels[i]

        self.projection = Projection(in_channels=prev_out, out_channels=out_channels, hidden_channels=projection_channels,
                                        non_linearity=non_linearity, n_dim=self.n_dim)
     
    def forward(self, x):
        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        skip_outputs = {}
        for layer_idx in range(self.n_layers):

            if layer_idx in  self.horizontal_skips_map.keys():
                #print("using skip", layer_idx)
                skip_val = skip_outputs[self.horizontal_skips_map[layer_idx]]
                output_scaling_factors = [m/n for (m,n) in zip(x.shape,skip_val.shape)]
                output_scaling_factors = output_scaling_factors[-1*self.n_dim:]
                t = resample(skip_val,output_scaling_factors, list(range(-self.n_dim, 0)))
                x = torch.cat([x,t], dim = 1)

            x = self.fno_blocks[layer_idx](x)

            if layer_idx in self.horizontal_skips_map.values():
                #print("saving skip", layer_idx)
                skip_outputs[layer_idx] = self.horizontal_skips[str(layer_idx)](x)

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x

UNO =  partialclass('UNO', UNO, factorization='Tucker')