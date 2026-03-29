"""
Using `torchtnt` to count FLOPS
================================

A demo using ``torchtnt`` to estimate the number of floating-point
operations per second (FLOPS) required for a model's forward and backward pass. 

This tutorial demonstrates how to profile neural operator models to understand
their computational requirements. FLOPS counting is crucial for:
- Comparing different model architectures
- Understanding computational bottlenecks
- Optimizing model efficiency
- Making informed decisions about model deployment

We will use the FLOP computation to analyze the computational resources
used by a FNO model.

"""

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Import dependencies
# -------------------
# We import the necessary modules for FLOPS counting and model creation

from copy import deepcopy
import torch
from torchtnt.utils.flops import FlopTensorDispatchMode

from neuralop.models import FNO

device = "cpu"

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Creating the FNO model for analysis
# ------------------------------------
# We create a moderately-sized FNO model to demonstrate FLOPS counting
fno = FNO(
    n_modes=(64, 64),
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    projection_channel_ratio=1,
)

# Create a sample input tensor for FLOPS counting
batch_size = 4
model_input = torch.randn(batch_size, 1, 128, 128)


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Counting FLOPS for forward and backward passes
# ----------------------------------------------
# We use the FlopTensorDispatchMode to count FLOPS during both forward and backward passes
with FlopTensorDispatchMode(fno) as ftdm:
    # Count forward pass FLOPS
    res = fno(model_input).mean()
    fno_forward_flops = deepcopy(ftdm.flop_counts)

    # Reset the counter and count backward pass FLOPS
    ftdm.reset()
    res.backward()
    fno_backward_flops = deepcopy(ftdm.flop_counts)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Analyzing FLOPS breakdown
# --------------------------
# The output is organized as a defaultdict object that counts the FLOPS used in each submodule.
# This gives us detailed insight into which parts of the model are computationally expensive.
print("Forward pass FLOPS breakdown:")
print(fno_forward_flops)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Finding maximum FLOPS usage
# ----------------------------
# To check the maximum FLOPS used during the forward pass, let's create a recursive function
# to search the nested dictionary structure:
from collections import defaultdict


def get_max_flops(flop_count_dict, max_value=0):
    for _, value in flop_count_dict.items():
        # If not nested, compare leaf value to max
        if isinstance(value, int):
            max_value = max(max_value, value)

        # Otherwise compute recursive max value below node
        elif isinstance(value, defaultdict):
            new_val = get_max_flops(value, max_value)
            max_value = max(max_value, new_val)
    return max_value


print(f"Max FLOPS required for FNO.forward: {get_max_flops(fno_forward_flops)}")
print(f"Max FLOPS required for FNO.backward: {get_max_flops(fno_backward_flops)}")

# %%
#
