import torch
import torch.nn as nn


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
    
    def set_embedding(self, x):
        self.embedding = x.reshape(self.embed_dim,)

    def forward(self, x):
        assert self.embedding is not None, "AdaIN: update embeddding before running forward"

        weight, bias = torch.split(self.mlp(self.embedding), self.in_channels, dim=0)

        return nn.functional.group_norm(x, self.in_channels, weight, bias, eps=self.eps)

class FlattenedInstanceNorm1d(nn.Module):
    def __init__(self, num_features: int):
        """FlattenedInstanceNorm1d takes 2d or greater dim
            tensors, flattens all data dimensions past 1st along the 1st data dim,
            applies 1d instance norm and reshapes.


        Parameters
        ----------
        num_features : int
            number of channels in instance norm
        """
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features=num_features)
    
    def forward(self, x):
        size = x.shape
        x = x.view(*size[:2], -1) # flatten everything past 3rd dim
        x = self.norm(x)
        # un-flatten last dims
        x = x.view(size[0], self.norm.num_features, *size[2:])
        return x