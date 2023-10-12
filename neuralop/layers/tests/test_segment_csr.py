import torch

from ..segment_csr import segment_csr

def test_segment_csr_backward():
    splits = torch.cumsum(torch.randint(0,10,(20,)), 0)
    data = torch.randn((splits[-1],3))

    out = segment_csr(data, splits, reduce='sum', use_scatter=False)
    loss = out.sum()
    loss.backward()

                            
                            