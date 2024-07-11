"""
Data Processors
=============================

In this example, we demonstrate how to use neuralop.data.transforms.DataProcessor
to preprocess and postprocess the small Darcy Flow example we ship with the package
for downstream use in training a neural operator model. 
"""

# %%
# 
import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

device = 'cpu'

"""
First, let's load the small Darcy Flow dataset:
"""
# %%
# Loading the Navier-Stokes dataset in 128x128 resolution
train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=32, 
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
        data_root="../neuralop/data/datasets/data/"
)

# %%
"""
Next let's visualize the data in its raw form.
"""