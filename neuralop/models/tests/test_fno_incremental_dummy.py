"""
Training a neural operator on Darcy-Flow - Author Robert Joseph
========================================
In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package
"""

# %%
# 

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO, FNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss
from tltorch import FactorizedTensor
from neuralop.models.fno_block import (FactorizedSpectralConv3d, FactorizedSpectralConv2d,
                                       FactorizedSpectralConv1d, FactorizedSpectralConv)
device = 'cpu'

# %%

def test_incremental_model_training(incremental_loss_gap=False, incremental=False, incremental_resolution=False):
    """_summary_

    Args:
            incremental_loss_gap (bool, optional): Loss gap method. Defaults to False.
            incremental (bool, optional): Gradient explained method. Defaults to False.
            incremental_resolution (bool, optional): Increase the resolution dynamically. Defaults to False.
    """        
    # DATASET
    # Loading the Darcy flow dataset
    
    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    incremental_mode = incremental_loss_gap or incremental
    baseline = incremental_mode or incremental_resolution

    if incremental_loss_gap:
        print('### INCREMENTAL LOSS GAP ###')
    if incremental:
        print('### INCREMENTAL ###')
    if incremental_resolution:
        print('### INCREMENTAL RESOLUTION ###')
    if not baseline:
        print('### BASELINE ###')
                    
    # Set up the incremental FNO model
    if incremental_mode:
        starting_modes = (2, 2)
    else:
        starting_modes = (16, 16)

    # set up model
    modes = (8, 8, 6, 6)
    incremental_modes = (2, 2, 4, 4)

    for dim in [1]:
        conv = FactorizedSpectralConv(2, 2, modes[:dim], n_layers=1, scale='auto', bias=False)

        original_weights = conv.weight[0].to_tensor().clone()
        print("Shape", original_weights.shape)
        print("Initial weights:", original_weights)
        #for name, param in conv.named_parameters():
        #    print(name, param.data)
            
        x = torch.randn(2, 2, *(8, )*dim)
        y = torch.randn(2, 2, *(8, )*dim)

        res = conv(x)
        loss = res.sum()
        loss.backward()
        
        # define a loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(conv.parameters(), lr=0.01)

        # run the input data through the model and update the weights
        
         # Dynamically reduce the number of modes in Fourier space
        conv.incremental_n_modes = incremental_modes[:dim]
        for i in range(20):
            res = conv(x)
            loss = criterion(res, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print the final weights of the model
        new_weights = conv.weight[0].to_tensor().clone()
        print("\nFinal weights:", new_weights)
        #for name, param in conv.named_parameters():
        #    print(name, param.data)
        
        #torch.testing.assert_close(original_weights, new_weights)
            
# toy very few incremental blocks - 
# create copy of weights
# full backard pass
# check only backward gradients of the incremental blocks are different
# add all the datasets as loaders

# Test Baseline Model first
#test_incremental_model_training(incremental_loss_gap=False, incremental=False, incremental_resolution=False)

# Test Incremental Loss Gap
test_incremental_model_training(incremental_loss_gap=True, incremental=False, incremental_resolution=False)

# Test Incremental
#test_incremental_model_training(incremental_loss_gap=False, incremental=True, incremental_resolution=False)

# Test Incremental Resolution
#test_incremental_model_training(incremental_loss_gap=False, incremental=False, incremental_resolution=True)



# %%
