import math

import torch
from torch import nn
import torch.nn.functional as F
from tltorch import TensorizedTensor
from tltorch.utils import get_tensorized_shape


class MLP(nn.Module):
    """A Multi-Layer Perceptron, with arbitrary number of layers
    
    Parameters
    ----------
    in_channels : int
    out_channels : int, default is None
        if None, same is in_channels
    hidden_channels : int, default is None
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : default is F.gelu
    dropout : float, default is 0
        if > 0, dropout probability
    """
    def __init__(self, in_channels, out_channels=None, hidden_channels=None, 
                 n_layers=2, n_dim=2, non_linearity=F.gelu, dropout=0., **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels 
        self.non_linearity = non_linearity
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)]) if dropout > 0. else None
        
        Conv = getattr(nn, f'Conv{n_dim}d')
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.fcs.append(Conv(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(Conv(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(Conv(self.hidden_channels, self.hidden_channels, 1))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout(x)

        return x

#Reimplementation of the MLP class using Linear instead of Conv 
class MLPLinear(torch.nn.Module):
    def __init__(self, layers, non_linearity=F.gelu, dropout=0.):
        super().__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.fcs = nn.ModuleList()
        self.non_linearity = non_linearity
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(self.n_layers)]) if dropout > 0. else None

        for j in range(self.n_layers):
            self.fcs.append(nn.Linear(layers[j], layers[j + 1]))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout(x)

        return x
    


class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
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


class AdaIN(nn.Module):
    def __init__(self, embed_dim, in_channels, mlp=None, eps=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.eps = eps

        if mlp is None:
            mlp = nn.Sequential(
                nn.Linear(embed_dim, 512),
                nn.GELU(),
                nn.Linear(512, 2*in_channels)
            )
        self.mlp = mlp

        self.embedding = None
    
    def update_embeddding(self, x):
        self.embedding = x.reshape(self.embed_dim,)

    def forward(self, x):
        assert self.embedding is not None, "AdaIN: update embeddding before running forward"
        
        weight, bias = torch.split(self.mlp(self.embedding), self.in_channels, dim=0)

        return nn.functional.group_norm(x, self.in_channels, weight, bias, eps=self.eps)
