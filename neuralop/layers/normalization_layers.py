import torch
import torch.nn as nn


class AdaIN(nn.Module):
    """Adaptive Instance Normalization (AdaIN) layer for style transfer in neural operators.

    AdaIN performs instance normalization followed by adaptive scaling and shifting
    based on an external embedding vector. This allows for style transfer by
    modulating the output characteristics based on a conditioning signal.

    The layer first normalizes the input using instance normalization, then applies
    learned scaling (weight) and shifting (bias) parameters derived from an embedding
    vector through a multi-layer perceptron.

    Parameters
    ----------
    embed_dim : int
        Dimension of the embedding vector used for style conditioning
    in_channels : int
        Number of input channels to normalize
    mlp : nn.Module, optional
        Multi-layer perceptron that maps embedding to (weight, bias) parameters.
        Should output 2*in_channels values. If None, a default MLP is created
        with architecture: Linear(embed_dim, 512) -> GELU -> Linear(512, 2*in_channels)
    eps : float, optional
        Small value added to the denominator for numerical stability in normalization.
        Default is 1e-5.
    """

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
        """Set the embedding vector for style conditioning."""
        self.embedding = x.reshape(self.embed_dim,)

    def forward(self, x):
        """Apply adaptive instance normalization to the input tensor."""
        assert self.embedding is not None, "AdaIN: update embeddding before running forward"

        weight, bias = torch.split(self.mlp(self.embedding), self.in_channels, dim=0)

        return nn.functional.group_norm(x, self.in_channels, weight, bias, eps=self.eps)


class InstanceNorm(nn.Module):
    """Dimension-agnostic instance normalization layer for neural operators.

    InstanceNorm normalizes each sample in the batch independently, computing
    mean and variance across spatial dimensions for each sample and channel
    separately. This is useful when the statistical properties of each sample
    are distinct and should be treated separately.

    Parameters
    ----------
    **kwargs : dict, optional
        Additional parameters to pass to torch.nn.functional.instance_norm().
        Common parameters include:
        - eps : float, optional
            Small value added to the denominator for numerical stability.
            Default is 1e-5.
        - momentum : float, optional
            Value used for the running_mean and running_var computation.
            Default is 0.1.
        - use_input_stats : bool, optional
            If True, use input statistics. Default is True.
        - weight : torch.Tensor, optional
            Weight tensor for affine transformation. If None, no scaling applied.
        - bias : torch.Tensor, optional
            Bias tensor for affine transformation. If None, no bias applied.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        """Apply instance normalization to the input tensor."""
        size = x.shape
        x = torch.nn.functional.instance_norm(x, **self.kwargs)
        assert x.shape == size
        return x


class BatchNorm(nn.Module):
    """Dimension-agnostic batch normalization layer for neural operators.

    BatchNorm normalizes data across the entire batch, computing a single mean
    and standard deviation for all samples combined. This is the most common
    form of normalization and is effective when batch statistics are a good
    approximation of the overall data distribution.

    For dimensions > 3, the layer automatically flattens spatial dimensions
    and uses BatchNorm1d, as PyTorch doesn't implement batch norm for 4D+ tensors.

    Parameters
    ----------
    n_dim : int
        Spatial dimension of input data (e.g., 1 for 1D, 2 for 2D, 3 for 3D).
        Determined by FNOBlocks.n_dim. If n_dim > 3, spatial dimensions are
        flattened and BatchNorm1d is used.
    num_features : int
        Number of channels in the input tensor to be normalized
    **kwargs : dict, optional
        Additional parameters to pass to the underlying batch normalization layer.
        Common parameters include:
        - eps : float, optional
            Small value added to the denominator for numerical stability.
            Default is 1e-5.
        - momentum : float, optional
            Value used for the running_mean and running_var computation.
            Default is 0.1.
        - affine : bool, optional
            If True, apply learnable affine transformation. Default is True.
        - track_running_stats : bool, optional
            If True, track running statistics. Default is True.
    """

    def __init__(self, n_dim: int, num_features: int, **kwargs):
        super().__init__()
        self.n_dim = n_dim
        self.num_features = num_features
        self.kwargs = kwargs

        if self.n_dim <= 3:
            self.norm = getattr(torch.nn, f"BatchNorm{n_dim}d")(
                num_features=num_features, **kwargs
            )
        else:
            print(
                "Warning: torch does not implement batch norm for dimensions higher than 3.\
                  We manually flatten the spatial dimension of 4+D tensors to apply batch norm. "
            )
            self.norm = torch.nn.BatchNorm1d(num_features=num_features, **kwargs)

    def forward(self, x):
        """Apply batch normalization to the input tensor."""
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
