import torch
from .. import segment_csr as segment_csr_module
from ..segment_csr import segment_csr

import pytest


@pytest.mark.parametrize("batch_size", [1, 4])
def test_native_segcsr_shapes(batch_size):
    n_pts = 25
    n_channels = 5
    max_nbrhd_size = 7  # prevent degenerate cases in testing

    # tensor to reduce
    src = torch.randn((batch_size, n_pts, n_channels))

    # randomly generate index pointer tensor for CSR format
    nbrhd_sizes = [torch.tensor([0])]
    while sum(nbrhd_sizes) < n_pts:
        nbrhd_sizes.append(torch.randint(0, max_nbrhd_size + 1, size=(1,)))
        max_nbrhd_size = min(max_nbrhd_size, n_pts - sum(nbrhd_sizes))
    indptr = torch.cumsum(torch.tensor(nbrhd_sizes, dtype=torch.long), dim=0)
    if batch_size > 1:
        indptr = indptr.repeat([batch_size] + [1] * indptr.ndim)
    else:
        src = src.squeeze(0)
    out = segment_csr(src, indptr, reduction="sum", use_scatter=False)

    if batch_size == 1:
        assert out.shape == (len(indptr) - 1, n_channels)
    else:
        assert out.shape == (batch_size, indptr.shape[1] - 1, n_channels)


def test_native_segcsr_reductions():
    src = torch.ones([10, 3])
    indptr = torch.tensor([0, 3, 8, 10], dtype=torch.long)

    out_sum = segment_csr(src, indptr, reduction="sum", use_scatter=False)
    assert out_sum.shape == (3, 3)
    diff = out_sum - torch.tensor([[3, 5, 2]]).T * torch.ones([3, 3])
    assert not diff.nonzero().any()

    out_mean = segment_csr(src, indptr, reduction="mean", use_scatter=False)
    assert out_mean.shape == (3, 3)
    diff = out_mean - torch.ones([3, 3])
    assert not diff.nonzero().any()


@pytest.mark.skipif(
    not hasattr(torch, "segment_reduce"),
    reason="torch.segment_reduce requires a newer PyTorch version",
)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_native_segment_reduce_matches_csr_fallback(batch_size, reduction):
    src = torch.arange(batch_size * 4 * 2, dtype=torch.float32).reshape(
        batch_size, 4, 2
    )
    indptr = torch.tensor([0, 2, 2, 4], dtype=torch.long)
    if batch_size > 1:
        batched_indptr = indptr.repeat(batch_size, 1)
        native_src = src
        native_indptr = batched_indptr
    else:
        native_src = src.squeeze(0)
        native_indptr = indptr

    native = segment_csr(
        native_src, native_indptr, reduction=reduction, use_scatter=False
    )
    expected = []
    for start, end in zip(indptr[:-1], indptr[1:]):
        values = (
            native_src[..., start:end, :]
            if batch_size > 1
            else native_src[start:end]
        )
        if end == start:
            values = torch.zeros_like(native_src[..., :1, :]).sum(dim=-2)
        elif reduction == "mean":
            values = values.mean(dim=-2)
        else:
            values = values.sum(dim=-2)
        expected.append(values)
    expected = torch.stack(expected, dim=-2)

    torch.testing.assert_close(native, expected)


@pytest.mark.skipif(
    not hasattr(torch, "segment_reduce"),
    reason="torch.segment_reduce requires a newer PyTorch version",
)
def test_compiling_prefers_native_segment_reduce(monkeypatch):
    monkeypatch.setattr(segment_csr_module, "_is_compiling", lambda: True)
    native_called = False
    torch_segment_reduce = torch.segment_reduce

    def recording_segment_reduce(*args, **kwargs):
        nonlocal native_called
        native_called = True
        return torch_segment_reduce(*args, **kwargs)

    monkeypatch.setattr(torch, "segment_reduce", recording_segment_reduce)

    src = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    indptr = torch.tensor([0, 2, 2, 4], dtype=torch.long)
    result = segment_csr(src, indptr, reduction="mean", use_scatter=True)
    expected = torch.tensor([[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]])
    assert native_called
    torch.testing.assert_close(result, expected)
