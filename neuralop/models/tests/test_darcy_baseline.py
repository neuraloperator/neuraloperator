"""
Training a neural operator on Darcy-Flow
========================================
In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package
"""

# %%
# 


import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO, FNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.datasets import burgers
from neuralop.datasets import darcy
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss
from asyncio import open_unix_connection
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
device = 'cpu'


# %%
# Loading the Navier-Stokes dataset in 128x128 resolution
#TRAIN_PATH = '/home/robert/data/burgers_data_R10.mat'
TRAIN_PATH = '/workspace/fly-incremental/data/fno/piececonst_r241_N1024_smooth1.mat'
TEST_PATH = '/home/robert/data/piececonst_r241_N1024_smooth2.mat'

train_loader, test_loader, output_encoder = darcy.load_darcy_pt(TRAIN_PATH, 800, [200]
                                                ,train_resolution=241, test_resolutions=[241], batch_size=32, test_batch_sizes=[32], positional_encoding=True, encode_input=False, encode_output=True, encoding='channel-wise', channel_dim=1)

# %%
# We create a FNO model
model = FNO(n_modes=(30, 30), hidden_channels=32, projection_channels=64)
model = model.to(device)
n_params = count_params(model)
print(f'\n Baseline model has {n_params} parameters.')
sys.stdout.flush()
#Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=1e-3, 
                                weight_decay=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}
print('\n### INCREMENTAL MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()

# Create the trainer for incremental model 
trainer = Trainer(model, n_epochs=500,
                  device=device,
                  mg_patching_levels=0,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True, incremental_loss_gap = False, incremental=False, incremental_resolution=False)

# Actually train the model on our small Darcy-Flow dataset
trainer.train(train_loader, test_loader,
              output_encoder,
              model, 
              optimizer,
              scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)