from typing import Literal
import importlib

import torch
from torch import einsum


def _is_compiling():
    """Return whether this function is being traced by ``torch.compile``."""
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    if is_compiling is not None:
        return is_compiling()

    # ``torch.compiler.is_compiling`` is not available on older supported
    # PyTorch versions. Keep the fallback private so importing this module does
    # not require a particular PyTorch release.
    dynamo = getattr(torch, "_dynamo", None)
    is_compiling = getattr(dynamo, "is_compiling", None)
    return is_compiling is not None and is_compiling()


def _native_segment_reduce(src, indptr, reduction):
    """Reduce CSR segments with the native PyTorch implementation."""
    batched = src.ndim == 3
    axis = 1 if batched else 0

    # ``segment_reduce`` returns NaN for empty mean segments. Dividing the sum
    # by clamped lengths preserves the historical zero output for empty
    # neighborhoods while retaining a single compiled native operation.
    out = torch.segment_reduce(src, "sum", offsets=indptr, axis=axis)
    if reduction == "mean":
        lengths = indptr.diff(dim=-1).to(dtype=out.dtype).clamp_min(1)
        out = out / lengths.unsqueeze(-1)
    return out


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
    properly built, segment_csr uses native PyTorch reduction when available,
    otherwise falling back to the legacy PyTorch implementation.

    When available, native ``torch.segment_reduce`` is used whenever
    ``torch.compile`` is tracing this function or ``use_scatter`` is False.
    This keeps compiled GINO models free of an opaque ``torch_scatter`` call,
    while preserving the faster ``torch_scatter`` path for eager execution.
    Older PyTorch versions without ``torch.segment_reduce`` retain the
    existing PyTorch fallback.

    Parameters
    ----------
    src : torch.Tensor
        tensor of features for each point
    indptr : torch.Tensor
        splits representing start and end indices
        of each neighborhood in src
    reduction : Literal['mean', 'sum']
        How to reduce a neighborhood. Options: 'mean', 'sum'. If mean,
        reduce by taking the average of all neighbors. Otherwise take the sum.
    use_scatter : bool, optional
        Whether to use torch-scatter.segment_csr for eager execution. If False,
        use native PyTorch reduction when available, by default True

        .. warning::

            torch-scatter is an optional dependency that conflicts with the newest versions of PyTorch,
            so you must handle the conflict explicitly in your environment. See :ref:`torch_scatter_dependency`
            for more information.
    """
    if reduction not in ["mean", "sum"]:
        raise ValueError("reduce must be one of 'mean', 'sum'")

    use_native = not use_scatter or _is_compiling()
    if use_native and hasattr(torch, "segment_reduce"):
        return _native_segment_reduce(src, indptr, reduction)

    if importlib.util.find_spec("torch_scatter") is not None and use_scatter:
        """only import torch_scatter when cuda is available"""
        import torch_scatter.segment_csr as scatter_segment_csr

        return scatter_segment_csr(src, indptr, reduce=reduction)

    else:
        if use_scatter:
            print(
                "Warning: use_scatter is True but torch_scatter is not properly built. \
                  Defaulting to naive PyTorch implementation"
            )
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
                from_idx = (slice(None), slice(indptr[0, i], indptr[0, i + 1]))
                ein_str = "bio->bo"
                start = indptr[0, i]
                n_nbrs = indptr[0, i + 1] - start
                to_idx = (slice(None), i)
            else:
                from_idx = slice(indptr[i], indptr[i + 1])
                ein_str = "io->o"
                start = indptr[i]
                n_nbrs = indptr[i + 1] - start
                to_idx = i
            src_from = src[from_idx]
            if n_nbrs > 0:
                to_reduce = einsum(ein_str, src_from)
                if reduction == "mean":
                    to_reduce /= n_nbrs
                out[to_idx] += to_reduce
        return out
