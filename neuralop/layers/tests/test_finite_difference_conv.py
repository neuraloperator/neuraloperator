####### ####### #######
# This script discretizes a parabola on a 2d grid and check to see that 
# the output converges to the theoretical limit.
####### ####### #######

import pytest
import numpy as np
import torch
import math

from ..differential_conv import FiniteDifferenceConvolution

def get_grid(S, batchsize, device):
    gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridx = gridx.reshape(1, 1, S, 1).repeat([batchsize, 1, 1, S])
    gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridy = gridy.reshape(1, 1, 1, S).repeat([batchsize, 1, S, 1])
    return torch.cat((gridx, gridy), dim=1).to(device)

@pytest.mark.parametrize('resolution', [500, 700, 1000])
def test_convergence_FiniteDifferenceConvolution_subtract_middle(resolution):
    torch.manual_seed(0)
    device = torch.device('cpu')

    num_channels = 10
    kernel_size = 3
    coeff = torch.rand((num_channels,))

    differential_block = FiniteDifferenceConvolution(
        in_channels=num_channels,
        out_channels=1,
        n_dim=2,
        kernel_size=kernel_size,
        groups=1,                       # mixing derivatives
    ).to(device)

    with torch.no_grad():
        weight = differential_block.conv.weight[0] # [0] because output channels is 1

    diff_block_output_list = []
    grid_width = 1 / resolution
    grid = get_grid(resolution, 1, device)
    
    channels = [torch.sum(coeff[i] * torch.square(grid), dim=1) for i in range(num_channels)]
    parabola = torch.stack(channels, dim=1).to(device)
    
    diff_block_output = differential_block(parabola, grid_width)
    diff_block_output_list.append(diff_block_output)

    theoretical_value = 0
    for k in range(num_channels):
        direction_k = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                direction_k += weight[k, i, j] * torch.tensor([[[[i - kernel_size // 2, j - kernel_size // 2]]]]).to(device)
        direction_k = direction_k.movedim(-1, 1).repeat([1, 1, resolution, resolution])
        theoretical_value += 2 * coeff[k] * torch.sum(grid * direction_k, dim=1)
    
    error = 1/(resolution-2) * torch.norm(diff_block_output.squeeze()[1:-1, 1:-1] - theoretical_value.squeeze()[1:-1, 1:-1]).item()

    assert(math.isclose(0, error, abs_tol=0.1))