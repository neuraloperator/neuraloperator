from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn


class Embedding(nn.Module, ABC):
    def __init__(self):
        super().__init__()
    
    @property
    @abstractmethod
    def out_channels(self):
        pass

class GridEmbedding2D(Embedding):
    """
    ``GridEmbedding2D`` applies a simple positional 
    embedding as a regular 2D grid. Expects inputs of shape
    ``(batch, channels, d_1, d_2)``

    Parameters
    ----------
    in_channels : ``int``
        number of channels in input. Fixed for output channel interface
    grid_boundaries : ``list``, optional
        coordinate boundaries of input grid, by default [[0, 1], [0, 1]]
    """
    def __init__(self, in_channels: int, grid_boundaries=[[0, 1], [0, 1]]):
        super().__init__()
        self.in_channels = in_channels
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None
    
    @property
    def out_channels(self):
        return self.in_channels + 2

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

class GridEmbeddingND(nn.Module):
    """
    GridEmbeddingND applies a simple positional 
    embedding as a regular ND grid. Expects inputs of shape 
    ``(batch, channels, d_1, ..., d_n)``.

    Parameters
    ----------
    in_channels : int
        number of channels in input
    dim : int
        dimensions of positional encoding to apply
    grid_boundaries : list, optional
        coordinate boundaries of input grid along each dim, by default [[0, 1], [0, 1]]
        """
    def __init__(self, in_channels: int, dim: int=2, grid_boundaries=[[0, 1], [0, 1]]):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        assert self.dim == len(grid_boundaries), f"Error: expected grid_boundaries to be\
            an iterable of length {self.dim}, received {grid_boundaries}"
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None
    
    @property
    def out_channels(self):
        return self.in_channels + self.dim

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
            assumes shape ``(batch (optional), channels, x_1, x_2, ...x_n)``
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

class SinusoidalEmbedding(Embedding):
    """
    SinusoidalEmbedding provides a unified sinusoidal positional embedding
    in the styles of Transformers [1]_ and Neural Radiance Fields (NERFs) [2]_.

    Expects inputs of shape ``(batch, n_in, in_channels)`` or ``(n_in, in_channels)``
    Parameters
    ----------
    in_channels : ``int``
        Number of input channels to embed
    num_freqs : ``int``, optional
        Number of frequencies in positional embedding.
        By default, set to the number of input channels
    embedding : ``{'transformer', 'nerf'}``
        Type of embedding to apply. For a function with N input channels, 
        each channel value p is embedded via a function g with 2L channels 
        such that g(p) is a 2L-dim vector. For 0 <= k < L:

        * ``'transformer'`` for transformer-style encoding.

            g(p)_k = sin((p / max_positions) ^ {k / N})

            g(p)_{k+1} = cos((p / max_positions) ^ {k / N})

        * ``'nerf'`` : NERF-style encoding.  

            g(p)_k = sin(2^(k) * Pi * p)

            g(p)_{k+1} = cos(2^(k) * Pi * p)

    max_positions : ``int``, optional
        Maximum number of positions for the encoding, default 10000
        Only used if `embedding == transformer`.

    References
    -----------
    .. [1] : Vaswani, A. et al (2017)
        "Attention Is All You Need". 
        NeurIPS 2017, https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf. 

    .. [2] : Mildenhall, B. et al (2020)
        "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis".
        ArXiv, https://arxiv.org/pdf/2003.08934. 
    """
    def __init__(self, 
                 in_channels: int,
                 num_frequencies: int=None, 
                 embedding_type: str='transformer', 
                 max_positions: int=10000):
        super().__init__()
        self.in_channels = in_channels
        self.num_frequencies = num_frequencies
        
        # verify embedding type
        allowed_embeddings = ['nerf', 'transformer']
        assert embedding_type in allowed_embeddings, \
            f"Error: embedding_type expected one of {allowed_embeddings}, received {embedding_type}"
        self.embedding_type = embedding_type
        if self.embedding_type == "transformer":
            assert max_positions is not None, "Error: max_positions must have an int value for \
                transformer embedding."
        self.max_positions = max_positions
    
    
    @property
    def out_channels(self):
        """
        required property for linking/composing model layers 
        """
        return 2 * self.num_frequencies * self.in_channels

    def forward(self, x):
        """
        Parameters 
        -----------
        x: ``torch.Tensor``
            shape ``(n_in, self.in_channels)`` or ``(batch, n_in, self.in_channels)``
        """
        assert x.ndim in [2,3], f"Error: expected inputs of shape (batch, n_in, {self.in_channels})\
            or (n_in, channels), got inputs with ndim={x.ndim}, shape={x.shape}"
        if x.ndim == 2:
            batched = False
            x = x.unsqueeze(0)
        else:
            batched = True
        batch_size, n_in, _ = x.shape
        
        if self.embedding_type == 'nerf':
            freqs = 2 ** torch.arange(0, self.num_frequencies, device=x.device) * torch.pi
        
        elif self.embedding_type == 'transformer':
            freqs = torch.arange(0, self.num_frequencies, device=x.device) / (self.num_frequencies * 2)
            freqs = (1 / self.max_positions) ** freqs
        
        # outer product of wavenumbers and position coordinates
        # shape b, n_in * channels, len(freqs)
        freqs = torch.einsum('bij, k -> bijk', x, freqs)

        # shape len(x), 2, len(freqs)
        freqs = torch.stack((freqs.sin(),freqs.cos()), dim=-1)

        # transpose the inner per-entry matrix and ravel to interleave sin and cos
        freqs = freqs.view(batch_size, n_in, -1)
        
        if not batched:
            freqs = freqs.squeeze(0)
        return freqs

class RotaryEmbedding2D(nn.Module):
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
        self.out_channels = 2

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
    
# Utility functions for GridEmbedding
def regular_grid_2d(spatial_dims, grid_boundaries=[[0, 1], [0, 1]]):
    """
    Creates a 2 x height x width stack of positional encodings A, where
    A[:,i,j] = [[x,y]] at coordinate (i,j) on a (height, width) grid. 
    """
    height, width = spatial_dims

    xt = torch.linspace(grid_boundaries[0][0], grid_boundaries[0][1],
                        height + 1)[:-1]
    yt = torch.linspace(grid_boundaries[1][0], grid_boundaries[1][1],
                        width + 1)[:-1]

    grid_x, grid_y = torch.meshgrid(xt, yt, indexing='ij')

    grid_x = grid_x.repeat(1, 1)
    grid_y = grid_y.repeat(1, 1)

    return grid_x, grid_y

def regular_grid_nd(resolutions: List[int], grid_boundaries: List[List[int]]=[[0,1]] * 2):
    """regular_grid_nd generates a tensor of coordinate points that 
    describe a bounded regular grid.
    
    Creates a dim x res_d1 x ... x res_dn stack of positional encodings A, where
    A[:,c1,c2,...] = [[d1,d2,...dn]] at coordinate (c1,c2,...cn) on a (res_d1, ...res_dn) grid. 

    Parameters
    ----------
    resolutions : List[int]
        resolution of the output grid along each dimension
    grid_boundaries : List[List[int]], optional
        List of pairs [start, end] of the boundaries of the
        regular grid. Must correspond 1-to-1 with resolutions default [[0,1], [0,1]]

    Returns
    -------
    grid: tuple(Tensor)
    list of tensors describing positional encoding 
    """
    assert len(resolutions) == len(grid_boundaries), "Error: inputs must have same number of dimensions"
    dim = len(resolutions)

    meshgrid_inputs = list()
    for res, (start,stop) in zip(resolutions, grid_boundaries):
        meshgrid_inputs.append(torch.linspace(start, stop, res + 1)[:-1])
    grid = torch.meshgrid(*meshgrid_inputs, indexing='ij')
    grid = tuple([x.repeat([1]*dim) for x in grid])
    return grid

  
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
