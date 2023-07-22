import torch
from torch import nn

try:
    import open3d.ml.torch as ml3d
    USE_OPEN3D_NEIGHBOR_SEARCH = True
except BaseException:
    import torch_cluster
    USE_OPEN3D_NEIGHBOR_SEARCH = False

class NeighborSearch(nn.Module):
    def __init__(self):
        super().__init__()
        if USE_OPEN3D_NEIGHBOR_SEARCH:
            self.nsearch = ml3d.layers.FixedRadiusSearch()

    def forward(self, data, queries, radius):
        return_dict = {}

        if USE_OPEN3D_NEIGHBOR_SEARCH:
            search_return = self.nsearch(data, queries, radius)
            return_dict['neighbors_index'] = search_return.neighbors_index.long()
            return_dict['neighbors_row_splits'] = search_return.neighbors_row_splits.long()

        else:
            neighbors_count, neighbors_index = torch_cluster.radius(data, queries, radius,
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