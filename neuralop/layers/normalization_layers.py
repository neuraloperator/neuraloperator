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

class InstanceNorm(nn.Module):
    def __init__(self, **kwargs):
        """InstanceNorm applies dim-agnostic instance normalization
        to data as an nn.Module. 

        kwargs: additional parameters to pass to instance_norm() for use as a module
        e.g. eps, affine
        """
        super().__init__()
        self.kwargs = kwargs
    
    def forward(self, x):
        size = x.shape
        x = torch.nn.functional.instance_norm(x, **self.kwargs)
        assert x.shape == size
        return x

class BatchNorm(nn.Module):
    def __init__(self, n_dim: int, num_features: int, **kwargs):
        """BatchNorm applies dim-agnostic batch normalization
        to data as an nn.Module. 

        Parameters
        ----------
        n_dim: int
            dimension of input data, determined by FNOBlocks.n_dim
            If n_dim > 3, batch norm will not be used.
        num_features: int
            number of channels in input to normalization 

        kwargs: additional parameters to pass to batch_norm() for use as a module
        e.g. eps, affine
        """
        super().__init__()
        self.n_dim = n_dim
        self.num_features = num_features
        self.kwargs = kwargs

        if self.n_dim <= 3:
            self.norm = getattr(torch.nn, f"BatchNorm{n_dim}d")(num_features=num_features, **kwargs)
        else:
            print("Warning: torch does not implement batch norm for dimensions higher than 3.\
                  We manually flatten the spatial dimension of 4+D tensors to apply batch norm. ")
            self.norm = torch.nn.BatchNorm1d(num_features=num_features, **kwargs)
    
    def forward(self, x):
        size = x.shape
        num_channels = size[1]

        # in 4+D, we flatten and use batchnorm1d.
        if self.n_dim >= 4:
            x = x.reshape(size[0], size[1], -1)
        x = self.norm(x)

        # if flattening occurred, unflatten
        if self.n_dim >= 4:
            x = x.reshape(size)
        
        assert x.shape == size
        return x