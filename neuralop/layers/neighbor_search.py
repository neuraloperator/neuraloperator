import torch
from torch import nn

#Requires either open3d torch instalation or torch_cluster
#Uses open3d by default which, as of 07/23/2023, requires torch 1.13.1
class NeighborSearch(nn.Module):
    """Neighbor search within a ball of a given radius

    Parameters
    ----------
    use_open3d : bool
        Wether to use open3d or torch_cluster
        NOTE: open3d implementation requires 3d data
    """
    def __init__(self, use_open3d=True):
        super().__init__()
        if use_open3d:
            from open3d.ml.torch.layers import FixedRadiusSearch
            self.search_fn = FixedRadiusSearch()
            self.use_open3d = use_open3d
        else:
            from torch_cluster import radius
            self.search_fn = radius
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
            neighbors_count, neighbors_index = self.search_fn(data, queries, radius,
                                               max_num_neighbors=data.shape[0])
            
            if neighbors_count[-1] != queries.shape[0] - 1:
                add_max_element = True
                neighbors_count = torch.cat((neighbors_count,
                                             torch.tensor([queries.shape[0] - 1], 
                                                          dtype=neighbors_count.dtype,
                                                          device=neighbors_count.device)), 
                                                          dim=0)
            else:
                add_max_element = False
            
            bins = torch.bincount(neighbors_count, minlength=1)
            if add_max_element:
                bins[-1] -= 1
            
            neighbors_row_splits = torch.cumsum(bins, dim=0)
            neighbors_row_splits = torch.cat((torch.tensor([0], 
                                              dtype=neighbors_row_splits.dtype,
                                              device=neighbors_row_splits.device),
                                              neighbors_row_splits), dim=0)
            

            return_dict['neighbors_index'] = neighbors_index.long()
            return_dict['neighbors_row_splits'] = neighbors_row_splits.long()
        
        return return_dict