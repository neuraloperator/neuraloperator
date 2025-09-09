from typing import List, Literal, Optional, Callable

import torch
from torch import nn
import torch.nn.functional as F

from .channel_mlp import LinearChannelMLP
from .integral_transform import IntegralTransform
from .neighbor_search import NeighborSearch
from .embeddings import SinusoidalEmbedding


class GNOBlock(nn.Module):
    """GNOBlock implements a Graph Neural Operator layer as described in [1]_.

    A GNO layer is a resolution-invariant operator that maps a function defined
    over one coordinate mesh to another defined over another coordinate mesh using 
    a pointwise kernel integral that takes contributions from neighbors of distance 1
    within a graph constructed via neighbor search with a specified radius. 

    Optionally, if provided, the input and output queries can have a positional embedding
    applied using the argument pos_embedding.

    The kernel integral computed in IntegralTransform 
    computes one of the following:
        (a) \\int_{A(x)} k(x, y) dy
        (b) \\int_{A(x)} k(x, y) * f(y) dy
        (c) \\int_{A(x)} k(x, y, f(y)) dy
        (d) \\int_{A(x)} k(x, y, f(y)) * f(y) dy
    
    Parameters
    ----------
    in_channels : int
        number of channels in input function. Only used if transform_type
        is (c) "nonlinear" or (d) "nonlinear_kernelonly"
    out_channels : int
        number of channels in output function
    coord_dim : int
        dimension of domain on which x and y are defined
    radius : float
        radius in which to search for neighbors
    weighting_fn : Callable, optional
        optional squared-norm weighting function to use for Mollified GNO layer
        by default None. See ``neuralop.layers.gno_weighting_functions` for more details. 
    reduction : Literal['sum', 'mean']
        whether to aggregate information from each neighborhood in the
        integral transform by summing (``'sum'``) or averaging (``'mean'``), by default ``'sum'``

    Other Parameters
    -----------------
    transform_type : str, optional
        Which integral transform to compute. The mapping is:
        'linear_kernelonly' -> (a)
        'linear' -> (b) [DEFAULT]
        'nonlinear_kernelonly' -> (c)
        'nonlinear' -> (d)
        If the input f is not given then (a) is computed
        by default independently of this parameter.
    pos_embedding_type: literal {'transformer', 'nerf'} | None
        type of positional embedding to use during the kernel integral transform.
        see `neuralop.layers.embeddings.SinusoidalEmbedding` for more details.
        default `'transformer'`
    pos_embedding_channels : int
        per-channel dimension of optional positional embedding to use, by default 32
    pos_embedding_max_positions: int
        `max_positions` parameter for SinusoidalEmbedding of type `'transformer'`. If
        `pos_embedding_type != 'transformer'`, this value is not used. Default 10000
    channel_mlp_layers : List[int], optional
        list of layer widths to dynamically construct
        LinearChannelMLP network to parameterize kernel k, by default None
    channel_mlp_non_linearity : torch.nn function, optional
        activation function for ChannelMLPLinear above, by default F.gelu
    channel_mlp : nn.Module, optional
        ChannelMLP parametrizing the kernel k. Input dimension
        should be dim x + dim y or dim x + dim y + dim f.
        ChannelMLP should not be pointwise and should only operate across
        channels to preserve the discretization-invariance of the 
        kernel integral. If you have more specific needs than the LinearChannelMLP,
        this argument allows you to pass your own Module to parameterize the kernel k. 
        Default None.
    use_torch_scatter_reduce : bool, optional
        whether to use ``torch-scatter`` to perform grouped reductions in the ``IntegralTransform``. 
        If False, uses native Python reduction in ``neuralop.layers.segment_csr``, by default True

        .. warning:: 

            ``torch-scatter`` is an optional dependency that conflicts with the newest versions of PyTorch,
            so you must handle the conflict explicitly in your environment. See :ref:`torch_scatter_dependency` 
            for more information. 
    use_open3d_neighbor_search : bool, optional
        whether to use open3d for fast 3d neighbor search, by default True. 
        
        .. note ::
            If the coordinates provided are not 3D, the ``GNOBlock`` automatically
            uses PyTorch native fallback neighbor search. 

    Examples
    ---------
    ```
    >>> gno = GNOBlock(in_channels=2, out_channels=12, coord_dim=3, radius=0.035)

    >>> gno
    GNOBlock(
        (pos_embedding): SinusoidalEmbedding()
        (neighbor_search): NeighborSearch()
        (integral_transform): IntegralTransform(
            (channel_mlp): LinearChannelMLP(
            (fcs): ModuleList(
                (0): Linear(in_features=384, out_features=128, bias=True)
                (1): Linear(in_features=128, out_features=256, bias=True)
                (2): Linear(in_features=256, out_features=128, bias=True)
                (3): Linear(in_features=128, out_features=12, bias=True)
            )
            )
        )
    )
    ```

    References
    -----------
    .. [1] : Zongyi Li, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, 
        Anima Anandkumar (2020). "Neural Operator: Graph Kernel Network for 
        Partial Differential Equations." ArXiV, https://arxiv.org/pdf/2003.03485.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 coord_dim: int,
                 radius: float,
                 transform_type="linear",
                 weighting_fn: Optional[Callable]=None,
                 reduction: Literal['sum', 'mean']='sum',
                 pos_embedding_type: str='transformer',
                 pos_embedding_channels: int=32,
                 pos_embedding_max_positions: int=10000,
                 channel_mlp_layers: List[int]=[128,256,128],
                 channel_mlp_non_linearity=F.gelu,
                 channel_mlp: nn.Module=None,
                 use_torch_scatter_reduce: bool=True,
                 use_open3d_neighbor_search: bool=True,):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.coord_dim = coord_dim

        self.radius = radius

        # Apply sinusoidal positional embedding
        self.pos_embedding_type = pos_embedding_type
        if self.pos_embedding_type in ['nerf', 'transformer']:
            self.pos_embedding = SinusoidalEmbedding(
                in_channels=coord_dim,
                num_frequencies=pos_embedding_channels,
                embedding_type=pos_embedding_type,
                max_positions=pos_embedding_max_positions
            )
        else:
            self.pos_embedding = None
                    
        # Create in-to-out nb search module
        if use_open3d_neighbor_search:
            assert self.coord_dim == 3, f"Error: open3d is only designed for 3d data, \
                GNO instantiated for dim={coord_dim}"
        self.neighbor_search = NeighborSearch(use_open3d=use_open3d_neighbor_search, return_norm=weighting_fn is not None)

        # create proper kernel input channel dim
        if self.pos_embedding is None:
            # x and y dim will be coordinate dim if no pos embedding is applied
            kernel_in_dim = self.coord_dim * 2
            kernel_in_dim_str = "dim(y) + dim(x)"
        else:
            # x and y dim will be embedding dim if pos embedding is applied
            kernel_in_dim = self.pos_embedding.out_channels * 2
            kernel_in_dim_str = "dim(y_embed) + dim(x_embed)"
            
        if transform_type == "nonlinear" or transform_type == "nonlinear_kernelonly":
            kernel_in_dim += self.in_channels
            kernel_in_dim_str += " + dim(f_y)"

        if channel_mlp is not None:
            assert channel_mlp.in_channels == kernel_in_dim, f"Error: expected ChannelMLP to take\
                  input with {kernel_in_dim} channels (feature channels={kernel_in_dim_str}),\
                      got {channel_mlp.in_channels}."
            assert channel_mlp.out_channels == out_channels, f"Error: expected ChannelMLP to have\
                 {out_channels=} but got {channel_mlp.in_channels=}."
            channel_mlp = channel_mlp

        elif channel_mlp_layers is not None:
            if channel_mlp_layers[0] != kernel_in_dim:
                channel_mlp_layers = [kernel_in_dim] + channel_mlp_layers
            if channel_mlp_layers[-1] != self.out_channels:
                channel_mlp_layers.append(self.out_channels)
            channel_mlp = LinearChannelMLP(layers=channel_mlp_layers, non_linearity=channel_mlp_non_linearity)

        # Create integral transform module
        self.integral_transform = IntegralTransform(
            channel_mlp=channel_mlp,
            transform_type=transform_type,
            use_torch_scatter=use_torch_scatter_reduce,
            weighting_fn=weighting_fn,
            reduction=reduction
        )

    def forward(self, y, x, f_y=None):
        """Compute a GNO neighbor search and kernel integral transform.

        Parameters
        ----------
        y : torch.Tensor of shape [n, d1]
            n points of dimension d1 specifying
            the space to integrate over.
            If batched, these must remain constant
            over the whole batch so no batch dim is needed.
        x : torch.Tensor of shape [m, d1], default None
            m points of dimension d1 over which the
            output function is defined. Must share domain
            with y
        f_y : torch.Tensor of shape [batch, n, d2] or [n, d2], default None
            Function to integrate the kernel against defined
            on the points y. The kernel is assumed diagonal
            hence its output shape must be d3 for the transforms
            (b) or (d). If None, (a) is computed.
        
        Output
        ----------
        out_features : torch.Tensor of shape [batch, m, d3] or [m, d3]
            Output function given on the points x.
            d4 is the output size of the kernel k.
        """
        if f_y is not None:
            if f_y.ndim == 3 and f_y.shape[0] == -1:
                f_y = f_y.squeeze(0)

        neighbors_dict = self.neighbor_search(data=y, queries=x, radius=self.radius)
        
        if self.pos_embedding is not None:
            y_embed = self.pos_embedding(y)
            x_embed = self.pos_embedding(x)
        else:
            y_embed = y
            x_embed = x

        out_features = self.integral_transform(y=y_embed,
                                               x=x_embed,
                                               neighbors=neighbors_dict,
                                               f_y=f_y)
        
        return out_features
