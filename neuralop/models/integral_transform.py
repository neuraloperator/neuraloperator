import torch
import torch.nn.functional as F

import open3d.ml.torch as ml3d

from torch import nn
from torch_scatter import segment_csr
from .mlp import MLPLinear


class NeighborSearch(nn.Module):
    def __init__(self):
        super().__init__()
        self.nsearch = ml3d.layers.FixedRadiusSearch()

    def forward(self, y, radius, x=None):
        if x is None:
            x = y
        
        return self.nsearch(y, x, radius)


class IntegralTransform(nn.Module):
    def __init__(self, mlp=None,
                 mlp_layers=None,
                 mlp_non_linearity=F.gelu, 
                 transform_type=0
                 ):
        
        super().__init__()

        assert mlp is not None or mlp_layers is not None
        assert transform_type >= 0 and transform_type <= 2

        self.transform_type = transform_type

        if mlp is None:
            self.mlp = MLPLinear(layers=mlp_layers,
                                 non_linearity=mlp_non_linearity)
        else:
            self.mlp = mlp
        
    """"
    Computes:
    \int_y k(x, y) [indepedent of type]
    \int_y k(x, y) * f(y) [linear transform, type: 0]
    \int_y k(x, y, f(y)) [non-linear transform, type: 1]
    \int_y k(x, y, f(y)) * f(y) [non-linear transform, type: 2]

    Assumes x=y if not specified
    Integral is taken w.r.t. the neighbors
    If no weights are given, a Monte-Carlo approximation is made
    NOTE: For transforms of type 0 or 2, out channels must be
    the same as the channels of f
    """
    def forward(self, y, neighbors,
                x=None, f_y=None, 
                weights=None):
        
        if x is None:
            x = y

        rep_features = y[neighbors.neighbors_index.long()]
        if f_y is not None:
            in_features = f_y[neighbors.neighbors_index.long()]

        rs = neighbors.neighbors_row_splits
        num_reps = rs[1:] - rs[:-1]
        self_features = torch.repeat_interleave(x, num_reps, dim=0)

        agg_features = torch.cat([rep_features, self_features], dim=1)
        if f_y is not None and (self.transform_type == 1 or self.transform_type == 2):
            agg_features = torch.cat([agg_features, in_features], dim=1)

        rep_features = self.mlp(agg_features)

        if f_y is not None and self.transform_type == 2:
            rep_features = rep_features*in_features 

        if weights is not None:
            rep_features = weights[neighbors.neighbors_index.long()]*rep_features
            reduction = 'sum'
        else:
            reduction = 'mean'

        out_features = segment_csr(rep_features, 
                                   neighbors.neighbors_row_splits, 
                                   reduce=reduction)
        return out_features