import torch
from ..segment_csr import segment_csr

import pytest

@pytest.mark.parametrize('batch_size', [1,4])
def test_native_segcsr(batch_size):
    n_pts = 25
    n_channels = 5
    max_nbrhd_size = 7 # prevent degenerate cases in testing

    # tensor to reduce
    src = torch.randn((batch_size, n_pts, n_channels))

    # randomly generate index pointer tensor for CSR format
    nbrhd_sizes = [torch.tensor([0])]
    while sum(nbrhd_sizes) < n_pts:
        nbrhd_sizes.append(torch.randint(0, max_nbrhd_size + 1, size=(1,)))
        max_nbrhd_size = min(max_nbrhd_size, n_pts - sum(nbrhd_sizes))
    indptr = torch.cumsum(torch.tensor(nbrhd_sizes, dtype=torch.long), dim=0)

    print(f"{indptr.shape=}")
    if batch_size > 1:
        indptr = indptr.repeat([batch_size] + [1]*indptr.ndim)
    else:
        src = src.squeeze(0)
    out = segment_csr(src=src, indptr=indptr, reduce='sum', use_scatter=False)
    
    if batch_size == 1:
        assert out.shape == (len(indptr) - 1, n_channels)
    else:
        assert out.shape == (batch_size, indptr.shape[1] - 1, n_channels)






