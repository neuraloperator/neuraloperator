import torch
from torch import nn

#Requires either open3d torch instalation or torch_cluster
#Uses open3d by default which, as of 07/23/2023, requires torch 1.13.1
class NeighborSearch(nn.Module):
    """Neighbor search within a ball of a given radius

    Parameters
    ----------
    use_open3d : bool
        Whether to use open3d or torch_cluster
        NOTE: open3d implementation requires 3d data
    """
    def __init__(self, use_open3d=True, use_torch_cluster=False):
        super().__init__()
        if use_open3d: # slightly faster, works on GPU in 3d only
            from open3d.ml.torch.layers import FixedRadiusSearch
            self.search_fn = FixedRadiusSearch()
            self.use_open3d = use_open3d
        else: # slower fallback, works on GPU and CPU
            from .simple_neighbor_search import simple_neighbor_search
            self.search_fn = simple_neighbor_search
            self.use_open3d = False
        
        
    def forward(self, data, queries, radius):
        """Find the neighbors, in data, of each point in queries
        within a ball of radius. Returns in CRS format.

        Parameters
        ----------
        data : torch.Tensor of shape [n, d]
            Search space of possible neighbors
            NOTE: open3d requires d=3
        queries : torch.Tensor of shape [m, d]
            Point for which to find neighbors
            NOTE: open3d requires d=3
        radius : float
            Radius of each ball: B(queries[j], radius)
        
        Output
        ----------
        return_dict : dict
            Dictionary with keys: neighbors_index, neighbors_row_splits
                neighbors_index: torch.Tensor with dtype=torch.int64
                    Index of each neighbor in data for every point
                    in queries. Neighbors are ordered in the same orderings
                    as the points in queries. Open3d and torch_cluster
                    implementations can differ by a permutation of the 
                    neighbors for every point.
                neighbors_row_splits: torch.Tensor of shape [m+1] with dtype=torch.int64
                    The value at index j is the sum of the number of
                    neighbors up to query point j-1. First element is 0
                    and last element is the total number of neighbors.
        """
        return_dict = {}

        if self.use_open3d:
            search_return = self.search_fn(data, queries, radius)
            return_dict['neighbors_index'] = search_return.neighbors_index.long()
            return_dict['neighbors_row_splits'] = search_return.neighbors_row_splits.long()

        else:
            return_dict = self.search_fn(data, queries, radius)
        
        return return_dict