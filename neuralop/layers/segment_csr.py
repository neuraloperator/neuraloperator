from typing import Literal
import importlib

import torch
from torch import einsum

def segment_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    reduction: Literal["mean", "sum"],
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
    reduce : Literal['mean', 'sum'], optional
        how to reduce a neighborhood. if mean,
        reduce by taking the average of all neighbors.
        Otherwise take the sum.
    use_scatter : bool, optional
        whether to use ``torch-scatter.segment_csr``. If False, uses native Python reduction.
        By default True

        .. warning:: 

            ``torch-scatter`` is an optional dependency that conflicts with the newest versions of PyTorch,
            so you must handle the conflict explicitly in your environment. See :ref:`torch_scatter_dependency` 
            for more information. 
    """
    if reduction not in ["mean", "sum"]:
        raise ValueError("reduce must be one of 'mean', 'sum'")

    if (
        importlib.util.find_spec("torch_scatter") is not None
        and use_scatter
    ):
        """only import torch_scatter when cuda is available"""
        import torch_scatter.segment_csr as scatter_segment_csr

        return scatter_segment_csr(src, indptr, reduce=reduction)

    else:
        if use_scatter:
            print("Warning: use_scatter is True but torch_scatter is not properly built. \
                  Defaulting to naive PyTorch implementation")
        # if batched, shape [b, n_reps, channels]
        # otherwise shape [n_reps, channels]
        if src.ndim == 3:
            batched = True
            point_dim = 1
        else:
            batched = False
            point_dim = 0

        # if batched, shape [b, n_out, channels]
        # otherwise shape [n_out, channels]
        output_shape = list(src.shape)
        n_out = indptr.shape[point_dim] - 1
        output_shape[point_dim] = n_out

        out = torch.zeros(output_shape, device=src.device)

        for i in range(n_out):
            # reduce all indices pointed to in indptr from src into out
            if batched:
                from_idx = (slice(None), slice(indptr[0,i], indptr[0,i+1]))
                ein_str = 'bio->bo'
                start = indptr[0,i]
                n_nbrs = indptr[0,i+1] - start
                to_idx = (slice(None), i)
            else:
                from_idx = slice(indptr[i], indptr[i+1])
                ein_str = 'io->o'
                start = indptr[i]
                n_nbrs = indptr[i+1] - start
                to_idx = i
            src_from = src[from_idx]
            if n_nbrs > 0:
                to_reduce = einsum(ein_str, src_from)
                if reduction == "mean":
                    to_reduce /= n_nbrs
                out[to_idx] += to_reduce
        return out
