import torch
from torch import nn
import torch.nn.functional as F

from .integral_transform import IntegralTransform
from .neighbor_search import NeighborSearch

class GNOBlock(nn.Module):
    """GNOBlock implements a Graph Neural Operator layer as described in [1]_.

    A GNO layer is a resolution-invariant operator that maps a function defined
    over one coordinate mesh to another defined over another coordinate mesh using 
    a pointwise kernel integral that takes contributions from neighbors of distance 1
    within a graph constructed via neighbor search with a specified radius. 
    
    Parameters
    ----------
    radius : float, optional
        _description_, by default None
    use_open3d_neighbor_search : _type_, optional
        _description_, by default None
    channel_mlp : _type_, optional
        _description_, by default None
    channel_mlp_layers : _type_, optional
        _description_, by default None
    channel_mlp_non_linearity : _type_, optional
        _description_, by default F.gelu
    transform_type : str, optional
        _description_, by default "linear"
    use_torch_scatter : bool, optional
        _description_, by default True

    References
    -----------
    _[1]. Neural Operator: Graph Kernel Network for Partial Differential Equations.
        Zongyi Li, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, 
        Anima Anandkumar. ArXiV, 2020 
    """
    def __init__(self,
                 radius: float=None,
                 n_layers: int=None,
                 channel_mlp=None,
                 channel_mlp_layers=None,
                 channel_mlp_non_linearity=F.gelu,
                 transform_type="linear",
                 use_open3d_neighbor_search: bool=None,
                 use_torch_scatter_reduce=True,):
        
        self.radius = radius
        self.n_layers = n_layers
        # Create in-to-out nb search module
        self.neighbor_search = NeighborSearch(use_open3d=use_open3d_neighbor_search)

        # Create integral transform module
        self.integral_transform = IntegralTransform(
            channel_mlp=channel_mlp,
            channel_mlp_layers=channel_mlp_layers,
            channel_mlp_non_linearity=channel_mlp_non_linearity,
            transform_type=transform_type,
            use_torch_scatter=use_torch_scatter_reduce
        )
    
    def forward(self, y, x, f_y=None, weights=None):
        """Compute a GNO neighbor search and kernel integral transform.

        Parameters
        ----------
        y : torch.Tensor of shape [n, d1]
            n points of dimension d1 specifying
            the space to integrate over.
            If batched, these must remain constant
            over the whole batch so no batch dim is needed.
        x : torch.Tensor of shape [m, d2], default None
            m points of dimension d2 over which the
            output function is defined.
        f_y : torch.Tensor of shape [batch, n, d3] or [n, d3], default None
            Function to integrate the kernel against defined
            on the points y. The kernel is assumed diagonal
            hence its output shape must be d3 for the transforms
            (b) or (d). If None, (a) is computed.
        weights : torch.Tensor of shape [n,], default None
            Weights for each point y proprtional to the
            volume around f(y) being integrated. For example,
            suppose d1=1 and let y_1 < y_2 < ... < y_{n+1}
            be some points. Then, for a Riemann sum,
            the weights are y_{j+1} - y_j. If None,
            1/|A(x)| is used.

        Output
        ----------
        out_features : torch.Tensor of shape [batch, m, d4] or [m, d4]
            Output function given on the points x.
            d4 is the output size of the kernel k.
        """
        neighbors_dict = self.neighbor_search(data=y, queries=x)
        out_features = self.integral_transform(y=y,
                                               x=x,
                                               neighbors=neighbors_dict,
                                               f_y=f_y)
        
        return out_features


