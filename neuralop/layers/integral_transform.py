import torch
from torch import nn
import torch.nn.functional as F

from .channel_mlp import LinearChannelMLP
from .segment_csr import segment_csr


class IntegralTransform(nn.Module):
    """Integral Kernel Transform (GNO)
    Computes one of the following:
        (a) \\int_{A(x)} k(x, y) dy
        (b) \\int_{A(x)} k(x, y) * f(y) dy
        (c) \\int_{A(x)} k(x, y, f(y)) dy
        (d) \\int_{A(x)} k(x, y, f(y)) * f(y) dy

    x : Points for which the output is defined

    y : Points for which the input is defined
    A(x) : A subset of all points y (depending on\
        each x) over which to integrate

    k : A kernel parametrized as a MLP (LinearChannelMLP)
    
    f : Input function to integrate against given\
        on the points y

    If f is not given, a transform of type (a)
    is computed. Otherwise transforms (b), (c),
    or (d) are computed. The sets A(x) are specified
    as a graph in CRS format.

    Parameters
    ----------
    channel_mlp : torch.nn.Module, default None
        MLP parametrizing the kernel k. Input dimension
        should be dim x + dim y or dim x + dim y + dim f.
        MLP should not be pointwise and should only operate across
        channels to preserve the discretization-invariance of the 
        kernel integral.
    channel_mlp_layers : list, default None
        List of layers sizes speficing a MLP which
        parametrizes the kernel k. The MLP will be
        instansiated by the LinearChannelMLP class
    channel_mlp_non_linearity : callable, default torch.nn.functional.gelu
        Non-linear function used to be used by the
        LinearChannelMLP class. Only used if channel_mlp_layers is
        given and channel_mlp is None
    transform_type : str, default 'linear'
        Which integral transform to compute. The mapping is:
        'linear_kernelonly' -> (a)
        'linear' -> (b)
        'nonlinear_kernelonly' -> (c)
        'nonlinear' -> (d)
        If the input f is not given then (a) is computed
        by default independently of this parameter.
    use_torch_scatter : bool, default 'True'
        whether to use ``torch-scatter`` to perform grouped reductions in the ``IntegralTransform``. 
        If False, uses native Python reduction in ``neuralop.layers.segment_csr``, by default True

        .. warning:: 

            ``torch-scatter`` is an optional dependency that conflicts with the newest versions of PyTorch,
            so you must handle the conflict explicitly in your environment. See :ref:`torch_scatter_dependency` 
            for more information. 
    """

    def __init__(
        self,
        channel_mlp=None,
        channel_mlp_layers=None,
        channel_mlp_non_linearity=F.gelu,
        transform_type="linear",
        weighting_fn=None,
        reduction='sum',
        use_torch_scatter=True,
    ):
        super().__init__()

        assert channel_mlp is not None or channel_mlp_layers is not None

        self.reduction = reduction
        self.transform_type = transform_type
        self.use_torch_scatter = use_torch_scatter
        if (
            self.transform_type != "linear_kernelonly"
            and self.transform_type != "linear"
            and self.transform_type != "nonlinear_kernelonly"
            and self.transform_type != "nonlinear"
        ):
            raise ValueError(
                f"Got transform_type={transform_type} but expected one of "
                "[linear_kernelonly, linear, nonlinear_kernelonly, nonlinear]"
            )

        if channel_mlp is None:
            self.channel_mlp = LinearChannelMLP(layers=channel_mlp_layers, non_linearity=channel_mlp_non_linearity)
        else:
            self.channel_mlp = channel_mlp
        
        self.weighting_fn = weighting_fn

    def forward(self, y, neighbors, x=None, f_y=None, weights=None):
        """Compute a kernel integral transform. Assumes x=y if not specified. 
        
        Integral is taken w.r.t. the neighbors.

        If no weights are given, a Monte-Carlo approximation is made.

        .. note :: For transforms of type 0 or 2, out channels must be
            the same as the channels of f

        Parameters
        ----------
        y : torch.Tensor of shape [n, d1]
            n points of dimension d1 specifying
            the space to integrate over.
            If batched, these must remain constant
            over the whole batch so no batch dim is needed.
        neighbors : dict
            The sets A(x) given in CRS format. The
            dict must contain the keys "neighbors_index"
            and "neighbors_row_splits." For descriptions
            of the two, see NeighborSearch.
            If batch > 1, the neighbors must be constant
            across the entire batch.
        x : torch.Tensor of shape [m, d2], default None
            m points of dimension d2 over which the
            output function is defined. If None,
            x = y.
        f_y : torch.Tensor of shape [batch, n, d3] or [n, d3], default None
            Function to integrate the kernel against defined
            on the points y. The kernel is assumed diagonal
            hence its output shape must be d3 for the transforms
            (b) or (d). If None, (a) is computed.
        weights : torch.Tensor of shape [n,], default None
            Weights for each point y proprtional to the
            volume around f(y) being integrated. For example,
            suppose d1=1 and let y_1 < y_2 < ... < y_{n+1}
            be some points. Then, for a Riemann sum,
            the weights are y_{j+1} - y_j. If None,
            1/|A(x)| is used.

        Output
        ----------
        out_features : torch.Tensor of shape [batch, m, d4] or [m, d4]
            Output function given on the points x.
            d4 is the output size of the kernel k.
        """

        if x is None:
            x = y

        rep_features = y[neighbors["neighbors_index"]]

        # batching only matters if f_y (latent embedding) values are provided
        batched = False
        # f_y has a batch dim IFF batched=True
        if f_y is not None:
            if f_y.ndim == 3:
                batched = True
                batch_size = f_y.shape[0]
                in_features = f_y[:, neighbors["neighbors_index"], :]
            elif f_y.ndim == 2:
                batched = False
                in_features = f_y[neighbors["neighbors_index"]]

        num_reps = (
            neighbors["neighbors_row_splits"][1:]
            - neighbors["neighbors_row_splits"][:-1]
        )

        self_features = torch.repeat_interleave(x, num_reps, dim=0)

        agg_features = torch.cat([rep_features, self_features], dim=-1)
        if f_y is not None and (
            self.transform_type == "nonlinear_kernelonly"
            or self.transform_type == "nonlinear"
        ):
            if batched:
                # repeat agg features for every example in the batch
                agg_features = agg_features.repeat(
                    [batch_size] + [1] * agg_features.ndim
                )
            agg_features = torch.cat([agg_features, in_features], dim=-1)

        rep_features = self.channel_mlp(agg_features)

        if f_y is not None and self.transform_type != "nonlinear_kernelonly":
            # if we have a batch of outputs (3d incl. batch dim) and unbatched inputs,
            # create an identical batch dim in rep_features
            if rep_features.ndim == 2 and batched:
                rep_features = rep_features.unsqueeze(0).repeat([batch_size] + [1] * rep_features.ndim)
            rep_features.mul_(in_features)
        
        # Weight neighbors in each neighborhood, first according to the neighbor search (mollified GNO)
        # and second according to individually-provided weights.
        nbr_weights = neighbors.get("weights")
        if nbr_weights is None:
            nbr_weights = weights
        if nbr_weights is None and self.weighting_fn is not None:
            raise KeyError("if a weighting function is provided, your neighborhoods must contain weights.")
        if nbr_weights is not None:
            nbr_weights = nbr_weights.unsqueeze(-1).unsqueeze(0)
            if self.weighting_fn is not None:
                nbr_weights = self.weighting_fn(nbr_weights)
            rep_features.mul_(nbr_weights)
            reduction = "sum" # Force sum reduction for weighted GNO layers

        else:
            reduction = self.reduction
        
        splits = neighbors["neighbors_row_splits"]
        if batched:
            splits = splits.unsqueeze(0).repeat([batch_size] + [1] * (splits.ndim))

        out_features = segment_csr(rep_features, splits, reduction=reduction, use_scatter=self.use_torch_scatter)
        return out_features
