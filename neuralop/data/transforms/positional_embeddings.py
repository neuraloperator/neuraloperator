from typing import List

import torch
from torch import nn

from .grid_transforms import regular_grid_2d, regular_grid_nd

class Euclidean2D(nn.Module):
    """A simple positional embedding as a regular 2D grid
    """
    def __init__(self, grid_boundaries=[[0, 1], [0, 1]]):
        """Euclidean2D applies a simple positional 
        embedding as a regular 2D grid

        Parameters
        ----------
        grid_boundaries : list, optional
            coordinate boundaries of input grid, by default [[0, 1], [0, 1]]
        """
        super().__init__()
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None

    def grid(self, spatial_dims, device, dtype):
        """grid generates 2D grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : torch.size
             sizes of spatial resolution
        device : literal 'cpu' or 'cuda:*'
            where to load data
        dtype : str
            dtype to encode data

        Returns
        -------
        torch.tensor
            output grids to concatenate 
        """
        # handle case of multiple train resolutions
        if self._grid is None or self._res != spatial_dims: 
            grid_x, grid_y = regular_grid_2d(spatial_dims,
                                      grid_boundaries=self.grid_boundaries)
            grid_x = grid_x.to(device).to(dtype).unsqueeze(0).unsqueeze(0)
            grid_y = grid_y.to(device).to(dtype).unsqueeze(0).unsqueeze(0)
            self._grid = grid_x, grid_y
            self._res = spatial_dims

        return self._grid

    def forward(self, data, batched=True):
        if not batched:
            if data.ndim == 3:
                data = data.unsqueeze(0)
        batch_size = data.shape[0]
        x, y = self.grid(data.shape[-2:], data.device, data.dtype)
        out =  torch.cat((data, x.expand(batch_size, -1, -1, -1),
                          y.expand(batch_size, -1, -1, -1)),
                         dim=1)
        # in the unbatched case, the dataloader will stack N 
        # examples with no batch dim to create one
        if not batched and batch_size == 1: 
            return out.squeeze(0)
        else:
            return out

class EuclideanND(nn.Module):
    """A positional embedding as a regular ND grid
    """
    def __init__(self, dim: int=2, grid_boundaries=[[0, 1], [0, 1]]):
        """EuclideanND applies a simple positional 
        embedding as a regular ND grid

        Parameters
        ----------
        dim: int
            dimensions of positional encoding to apply
        grid_boundaries : list, optional
            coordinate boundaries of input grid along each dim, by default [[0, 1], [0, 1]]
        """
        super().__init__()
        self.dim = dim
        assert self.dim == len(grid_boundaries), f"Error: expected grid_boundaries to be\
            an iterable of length {self.dim}, received {grid_boundaries}"
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None

    def grid(self, spatial_dims: torch.Size, device: str, dtype: torch.dtype):
        """grid generates ND grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : torch.Size
             sizes of spatial resolution
        device : literal 'cpu' or 'cuda:*'
            where to load data
        dtype : str
            dtype to encode data

        Returns
        -------
        torch.tensor
            output grids to concatenate 
        """
        # handle case of multiple train resolutions
        if self._grid is None or self._res != spatial_dims: 
            grids_by_dim = regular_grid_nd(spatial_dims,
                                      grid_boundaries=self.grid_boundaries)
            # add batch, channel dims
            grids_by_dim = [x.to(device).to(dtype).unsqueeze(0).unsqueeze(0) for x in grids_by_dim]
            self._grid = grids_by_dim
            self._res = spatial_dims

        return self._grid

    def forward(self, data, batched=True):
        """
        Params
        --------
        data: torch.Tensor
            assumes shape batch (optional), channels, x_1, x_2, ...x_n
        batched: bool
            whether data has a batch dim
        """
        # add batch dim if it doesn't exist
        if not batched:
            if data.ndim == self.dim + 1:
                data = data.unsqueeze(0)
        batch_size = data.shape[0]
        grids = self.grid(spatial_dims=data.shape[2:],
                          device=data.device,
                          dtype=data.dtype)
        grids = [x.repeat(batch_size, *[1] * (self.dim+1)) for x in grids]
        out =  torch.cat((data, *grids),
                         dim=1)
        return out
    
class Sinusoidal2D(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        """Sinusoidal2D applies a 2d sinusoidal positional encoding 

        Parameters
        ----------
        num_channels : int
            number of input channels
        max_positions : int, optional
            maximum positions to encode, by default 10000
        endpoint : bool, optional
            whether to set endpoint, by default False
        """
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
    
# Utility fucntions for Rotary embedding
# modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
def rotate_half(x):
    """
    Split x's channels into two equal halves.
    """
    # split the last dimension of x into two equal halves
    x = x.reshape(*x.shape[:-1], 2, -1)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    Apply rotation matrix computed based on freqs to rotate t.
    t: tensor of shape [batch_size, num_points, dim]
    freqs: tensor of shape [batch_size, num_points, 1]

    Formula: see equation (34) in https://arxiv.org/pdf/2104.09864.pdf
    """
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


class Rotary(nn.Module):
    def __init__(self, dim, min_freq=1/64, scale=1.):
        """
        Applying rotary positional embedding (https://arxiv.org/abs/2104.09864) to the input feature tensor.
        The crux is the dot product of two rotation matrices R(theta1) and R(theta2) is equal to R(theta2 - theta1).
        """
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, coordinates):
        """coordinates is tensor of [batch_size, num_points]"""
        coordinates = coordinates * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', coordinates, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]

    @staticmethod
    def apply_1d_rotary_pos_emb(t, freqs):
        return apply_rotary_pos_emb(t, freqs)

    @staticmethod
    def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
        """Split the last dimension of features into two equal halves
           and apply 1d rotary positional embedding to each half."""
        d = t.shape[-1]
        t_x, t_y = t[..., :d//2], t[..., d//2:]

        return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                          apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)