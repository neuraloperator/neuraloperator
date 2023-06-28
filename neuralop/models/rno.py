"""
Author: Miguel Liu-Schiaffini (mliuschi@caltech.edu)
    `FourierLayer2d` by Zongyi Li 

This file implements a recurrent neural operator (RNO) based on the gated recurrent unit (GRU)
and Fourier neural operator (FNO) architectures.

In particular, the RNO has an identical architecture to the finite-dimensional GRU, 
with the exception that linear matrix-vector multiplications are replaced by linear 
Fourier layers (see Li et al., 2021), and for regression problems, the output nonlinearity
is replaced with a SELU activation.

We call this model the 2D RNO, in the sense that it solves PDE trajectories and 
takes the FFT in time only and in one space dimension.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import operator
from functools import reduce
import scipy.io
import sys
sys.path.append('ks')
from libs.models.spectral_conv import SpectralConv2d
from libs.models.transformer_models import SpectralRegressor, SimpleTransformerEncoderLayer
from libs.utilities3 import *
from libs.models.attention_layers import PositionalEncoding, NeRFPosEmbedding

torch.manual_seed(0)
np.random.seed(0)


class FourierLayer2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FourierLayer2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.spec_conv = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.norm_conv1d = nn.Conv1d(self.width, self.width, 1)

    def forward(self, x):
        batch_size, dim, dom_size1, dom_size2 = x.shape
        x1 = self.spec_conv(x)
        x2 = self.norm_conv1d(x.reshape((batch_size, dim, dom_size1 * dom_size2))).view(batch_size, self.width, dom_size1, dom_size2)
        return x1 + x2


class RNO_cell(nn.Module):
    def __init__(self, in_dim, out_dim, modes1, modes2, width):
        super(RNO_cell, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.f1 = FourierLayer2d(self.modes1, self.modes2, self.width)
        self.f2 = FourierLayer2d(self.modes1, self.modes2, self.width)
        self.f3 = FourierLayer2d(self.modes1, self.modes2, self.width)
        self.f4 = FourierLayer2d(self.modes1, self.modes2, self.width)
        self.f5 = FourierLayer2d(self.modes1, self.modes2, self.width)
        self.f6 = FourierLayer2d(self.modes1, self.modes2, self.width)
        self.f7 = FourierLayer2d(self.modes1, self.modes2, self.width)
        self.f8 = FourierLayer2d(self.modes1, self.modes2, self.width)

        self.b1 = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.)))
        self.b2 = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.)))
        self.b3 = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.)))
        self.b4 = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.)))
    
    def forward(self, x, h):
        z = torch.sigmoid(self.f1(x) + self.f2(h) + self.b1)
        z2 = torch.sigmoid(self.f7(x) + self.f8(h) + self.b4)
        r = torch.sigmoid(self.f3(x) + self.f4(h) + self.b2)
        h_hat = F.selu(self.f5(x) + self.f6(r * h) + self.b3)
        h_next = (1. - z) * h + z2 * h_hat
        return h_next

    
class RNO_layer(nn.Module):
    def __init__(self, in_dim, out_dim, modes1, modes2, width, return_sequences=False):
        super(RNO_layer, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.return_sequences = return_sequences
        self.cell = RNO_cell(in_dim, out_dim, modes1, modes2, width)
        self.bias_h = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.)))

    def forward(self, x, h=None):
        batch_size, timesteps, dim, dom_size1, dom_size2 = x.shape

        if h is None:
            h = torch.zeros((batch_size, self.width, dom_size1, dom_size2)).to(x.device)
            h += self.bias_h
        outputs = []
        for i in range(timesteps):
            h = self.cell(x[:, i], h)
            if self.return_sequences:
                outputs.append(h)

        if self.return_sequences:
            return torch.stack(outputs, dim=1)
        else:
            return h


class RNO2dObserverOld(nn.Module):
    def __init__(self, modes1, modes2, width, recurrent_index, layer_num=3, pad_amount=None, pad_dim='1'):
        """
            `pad_dim` can be '1', '2', or 'both', and this decides which of the two space dimensions to pad
            `pad_amount` is a tuple that determines how much to pad each dimension by, if `pad_dim`
            specifies that dimension should be padded.
        """
        super(RNO2dObserverOld, self).__init__()
        self.modes1 = modes1
        self.modes1 = modes2
        self.width = width
        self.pad_amount = pad_amount # pads dom_size1 dimension
        self.pad_dim = pad_dim
        self.recurrent_index = recurrent_index
        data_in_dim, pos_in_dim = 1, 0
        self.in_dim = data_in_dim + pos_in_dim
        self.out_dim = 1
        self.layer_num = layer_num
        self.input_projection_layer = nn.Linear(self.in_dim, self.width) # input channel_dim is in_dim + 1: u is in-dim, and grid is 2 dim
        torch.nn.init.normal_(self.input_projection_layer.weight, mean=0, std=1)
        module_list = [RNO_layer(self.width, self.width, modes1, modes2, width, return_sequences=True)
                                     for _ in range(layer_num - 1)]
        module_list.append(RNO_layer(self.width, self.width, modes1, modes2, width, return_sequences=False))
        self.layers = nn.ModuleList(module_list)
        self.regressor = SpectralRegressor(in_dim=self.width, n_hidden=self.width, freq_dim=self.width, 
                                           out_dim=self.out_dim, modes=modes2, activation='relu', dropout=0.3)
    
    def forward_one_step(self, x, v_plane=None, init_hidden_states=None): # h must be padded if using padding
        if init_hidden_states is None:
            init_hidden_states = [None] * self.layer_num
        batch_size, timesteps, dom_size1, dom_size2, dim = x.shape
        x = self.input_projection_layer(x)
        x = x.permute(0, 1, 4, 2, 3) # new shape: (batch, timesteps, dim, dom_size1, dom_size2)
        if self.pad_amount: # pad the domain if input is non-periodic
            if self.pad_dim == '1':
                x = x.permute(0, 1, 2, 4, 3) # new shape: (batch, timesteps, dim, dom_size2, dom_size1)
                x = F.pad(x, [0,self.pad_amount[0]])
                x = x.permute(0, 1, 2, 4, 3)
            elif self.pad_dim == '2':
                x = F.pad(x, [0,self.pad_amount[1]])
            elif self.pad_dim == 'both':
                x = x.permute(0, 1, 2, 4, 3) # new shape: (batch, timesteps, dim, dom_size2, dom_size1)
                x = F.pad(x, [0,self.pad_amount[0]])
                x = x.permute(0, 1, 2, 4, 3)
                x = F.pad(x, [0,self.pad_amount[1]])
                
        final_hidden_states = []
        for i in range(self.layer_num):
            pred_x = self.layers[i](x, init_hidden_states[i])
            # using residual predictions
            if i < self.layer_num - 1:
                x = x + pred_x
                final_hidden_states.append(x[:, -1])
            else:
                x = pred_x
                final_hidden_states.append(x)
        h = final_hidden_states[-1]
        
        if self.pad_amount: # remove padding
            if self.pad_dim == '1':
                h = h[:, :, :-self.pad_amount[0]]
            elif self.pad_dim == '2':
                h = h[..., :-self.pad_amount[1]]
            elif self.pad_dim == 'both':
                h = h[:, :, :-self.pad_amount[0]]
                h = h[..., :-self.pad_amount[1]]
        
        h = h.permute(0, 2, 3, 1)
        pred = self.regressor(h)
        return pred, final_hidden_states

    def forward(self, x, v_plane=None, timestep=2):
        bs, timestep, xshape, yshape, dim = x.shape
        result = self.predict(x, num_steps=timestep)
        result = result[:, self.recurrent_index, :, :, :]
        return result

    def predict(self, x, num_steps): # num_steps is the number of steps ahead to predict
        output = []
        states = [None] * self.layer_num
        
        for i in range(num_steps):
            pred, states = self.forward_one_step(x, init_hidden_states=states)
            output.append(pred)
            x = pred.reshape((pred.shape[0], 1, pred.shape[1], pred.shape[2], pred.shape[3]))

        return torch.stack(output, dim=1)

    def get_grid(self, shape, device):
        batchsize, steps, dom_size1, dom_size2, _ = shape

        gridx = torch.tensor(np.linspace(0, 1, dom_size1), dtype=torch.float)
        gridx = gridx.reshape(1, 1, dom_size1, 1, 1).repeat([batchsize, steps, 1, dom_size2, 1])
        gridy = torch.tensor(np.linspace(0, 1, dom_size2), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, dom_size2, 1).repeat([batchsize, steps, dom_size1, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def count_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return int(sum([np.prod(p.size()) for p in model_parameters]))