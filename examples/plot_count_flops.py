"""
Using `torchtnt` to count FLOPS
=============================

In this example, we demonstrate how to use torchtnt to estimate the number of floating-point
operations per second (FLOPS) required for a model's forward and backward pass. 

We will use the FLOP computation to compare the resources used by a base FNO and a tensorized FNO. 
"""

# %%
# 
from copy import deepcopy
import torch
from torchtnt.utils.flops import FlopTensorDispatchMode

from neuralop.models import FNO, TFNO

device = 'cpu'

fno = FNO(n_modes=(64,64), 
          in_channels=3, 
          out_channels=1, 
          hidden_channels=64, 
          projection_channels=64)

tfno = TFNO(n_modes=(64,64), 
          in_channels=3, 
          out_channels=1, 
          hidden_channels=64, 
          projection_channels=64)

batch_size = 4
model_input = torch.randn(batch_size, 3, 128, 128)


with FlopTensorDispatchMode(fno) as ftdm:
    # count forward flops
    res = fno(model_input).mean()
    fno_forward_flops = deepcopy(ftdm.flop_counts)
    
    ftdm.reset()
    res.backward()
    fno_flops_backward = deepcopy(ftdm.flop_counts)
# %%
# N
print(fno_forward_flops)