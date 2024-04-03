from typing import Literal
import importlib

import torch


def segment_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    reduce: Literal["mean", "sum"],
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
    """
    if reduce not in ["mean", "sum"]:
        raise ValueError("reduce must be one of 'mean', 'sum'")

    if (
        torch.backends.cuda.is_built()
        and importlib.util.find_spec("torch_scatter") is not None
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
        if src.ndim == 3:
            batched = True
            point_dim = 1
            indices = indptr[0]  # full shape is needed for CUDA but not CPU-only
        else:
            batched = False
            point_dim = 0
            indices = indptr

        # end indices - start indices
        n_nbrs = indices[1:] - indices[:-1]

        output_shape = list(src.shape)
        output_shape[point_dim] = indices.shape[0] - 1

        out = torch.zeros(output_shape, device=src.device)

        for i, start in enumerate(indices[:-1]):
            if batched:
                out_idx = (slice(None), slice(i, i + 1))  # grab all batch elements
            else:
                out_idx = i  # no batching dim

            if (
                start == src.shape[point_dim]
            ):  # if the last neighborhoods are empty, skip
                break
            for j in range(n_nbrs[i]):
                if batched:
                    src_idx = (slice(None), slice(start + j, start + j + 1))
                else:
                    src_idx = start = j
                out[out_idx] += src[src_idx]
            if reduce == "mean":
                out[out_idx] /= n_nbrs[i]
        return out
