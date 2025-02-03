from typing import Literal
import importlib

import torch

def segment_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    reduce: Literal["mean", "sum"],
    eps: float=1e-7,
    use_scatter=True,
):
    """segment_csr reduces all entries of a CSR-formatted
    matrix by summing or averaging over neighbors.

    Used to reduce features over neighborhoods
    in neuralop.layers.IntegralTransform

    If use_scatter is set to False or torch_scatter is not
    properly built, segment_csr falls back to a naive PyTorch implementation

    Note: the native version is mainly intended for running tests on 
    CPU-only GitHub CI runners to get around a versioning issue. 
    torch_scatter should be installed and built if possible. 

    Parameters
    ----------
    src : torch.Tensor
        tensor of features for each point
    indptr : torch.Tensor
        splits representing start and end indices
        of each neighborhood in src
    reduce : Literal['mean', 'sum']
        how to reduce a neighborhood. if mean,
        reduce by taking the average of all neighbors.
        Otherwise take the sum.
    eps : float
        Tiny perturbation to prevent div by zero in scaling
        neigborhoods if a particular neighborhood is empty
    use_scatter : bool
        whether to use `torch_scatter`'s implementation of 
        `segment_csr` over the native torch version. Defaults to True
    """
    if reduce not in ["mean", "sum"]:
        raise ValueError("reduce must be one of 'mean', 'sum'")

    if (
        importlib.util.find_spec("torch_scatter") is not None
        and use_scatter
    ):
        """only import torch_scatter when cuda is available"""
        import torch_scatter.segment_csr as scatter_segment_csr

        return scatter_segment_csr(src, indptr, reduce=reduce)

    else:
        if use_scatter:
            print("Warning: use_scatter is True but torch_scatter is not properly built. \
                  Defaulting to naive PyTorch implementation")
        # if batched, shape [b, n_reps, channels]
        # otherwise shape [n_reps, channels]
        """segment_csr reduces all entries of a CSR-formatted
    matrix by summing or averaging over neighbors.

    Used to reduce features over neighborhoods
    in neuralop.layers.IntegralTransform

    If use_scatter is set to False or torch_scatter is not
    properly built, segment_csr falls back to a naive PyTorch implementation

    Note: the native version is mainly intended for running tests on 
    CPU-only GitHub CI runners to get around a versioning issue. 
    torch_scatter should be installed and built if possible. 

    Parameters
    ----------
    src : torch.Tensor
        tensor of features for each point
    indptr : torch.Tensor
        splits representing start and end indices
        of each neighborhood in src
    reduce : Literal['mean', 'sum']
        how to reduce a neighborhood. if mean,
        reduce by taking the average of all neighbors.
        Otherwise take the sum.
    eps : float
        Tiny perturbation to prevent div by zero in scaling
        neigborhoods if a particular neighborhood is empty
    use_scatter : bool
        whether to use `torch_scatter`'s implementation of 
        `segment_csr` over the native torch version. Defaults to True
    """

        if src.ndim == 3:
            point_dim = 1
        else:
            point_dim = 0

        # if batched, shape [b, n_out, channels]
        # otherwise shape [n_out, channels]    
        n_in = src.shape[point_dim]

        n_reps = indptr[1:] - indptr[:-1]

        inds = torch.arange(n_in).unsqueeze(0)
        mask = (inds >= indptr[:-1].unsqueeze(1)) & (inds < indptr[1:].unsqueeze(1))

        # add a batch dim to the mask if src is batched
        if src.ndim == 3:
            mask = mask.unsqueeze(0)
        out = (mask.unsqueeze(-1).to(src.dtype) * src.unsqueeze(point_dim)).sum(dim=point_dim+1)

        if reduce == 'mean':
            # scale the outputs by number of reduced neighbors, add eps to avoid div by zero
            scale = (1 / (n_reps + eps)).unsqueeze(0).T
            out = out * scale

        return out
