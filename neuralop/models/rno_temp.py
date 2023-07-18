"""
Author: Miguel Liu-Schiaffini (mliuschi@caltech.edu) and Zelin Zhao (sjtuytc@gmail.com)
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
from torch.nn.parameter import Parameter
from functools import partial
from torch.nn.init import xavier_normal_
#from .spectral_convolution import FactorizedSpectralConv2d


def default(value, d):
    '''
    helper taken from https://github.com/lucidrains/linear-attention-transformer
    '''
    return d if value is None else value

    
# class SpectralConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2, norm='ortho'):
#         super(SpectralConv2d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes1 = modes1
#         self.modes2 = modes2
        
#         scale = 1 / (in_channels * out_channels)
#         self.fourier_weight = nn.ParameterList([
#             nn.Parameter(torch.FloatTensor(in_channels, out_channels, modes1, modes2, 2))
#             for _ in range(2)
#         ])
#         for param in self.fourier_weight:
#             nn.init.xavier_normal_(param, gain=scale * np.sqrt(in_channels + out_channels))
#         self.norm = norm

#     @staticmethod
#     def complex_matmul_2d(a, b):
#         # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
#         op = partial(torch.einsum, "bixy,ioxy->boxy")
#         return torch.stack([
#             op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
#             op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
#         ], dim=-1)

#     def forward(self, x):
#         '''
#         Input: (-1, n_grid**2, in_features) or (-1, n_grid, n_grid, in_features)
#         Output: (-1, n_grid**2, out_features) or (-1, n_grid, n_grid, out_features)
#         '''
#         batchsize = x.shape[0]
#         n = x.shape[-1]
#         x_ft = torch.fft.rfft2(x, s=(n, n), norm=self.norm)
#         x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)

#         out_ft = torch.zeros(x.size(0), self.out_channels, n, n // 2 + 1, 2, device=x.device)
#         out_ft[:, :, :self.modes1, :self.modes2] = self.complex_matmul_2d(
#             x_ft[:, :, :self.modes1, :self.modes2], self.fourier_weight[0])
#         out_ft[:, :, -self.modes1:, :self.modes2] = self.complex_matmul_2d(
#             x_ft[:, :, -self.modes1:, :self.modes2], self.fourier_weight[1])
#         out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])
#         x = torch.fft.irfft2(out_ft, s=(n, n), norm=self.norm)
#         return x

def compl_mul2d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixy,ioxy->boxy", a, b)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, norm="ortho"):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.norm = norm

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        size1 = x.shape[-2]
        size2 = x.shape[-1]

        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2,3], norm=self.norm)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, size1, size2//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)


        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(size1, size2), dim=[2,3], norm=self.norm)
        return x


class SpectralConvWithFC(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, n_grid=None, dropout=0.1, norm='ortho',
                 activation='silu', return_freq=False, debug=False):
        super(SpectralConvWithFC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spec_conv = SpectralConv2d(in_channels, out_channels, modes1, modes2, norm)
        self.linear = nn.Linear(in_channels, out_channels)
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.return_freq = return_freq

    def forward(self, x):
        '''
        Input: (-1, n_grid, n_grid, in_features)
        Output: (-1, n_grid, n_grid, out_features)
        '''
        res = self.linear(x)
        x = self.dropout(x)
        x = x.permute(0, 3, 1, 2)
        x = self.spec_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.activation(x + res)
        if self.return_freq:
            raise RuntimeError("Not supported return freq")
        else:
            return x
        
    
class SpectralRegressor(nn.Module):
    def __init__(self, in_dim,
                 n_hidden,
                 freq_dim,
                 out_dim,
                 modes1: int,
                 modes2: int,
                 num_spectral_layers: int = 2,
                 n_grid=None,
                 dim_feedforward=None,
                 spacial_fc=False,
                 spacial_dim=2,
                 return_freq=False,
                 return_latent=False,
                 normalizer=None,
                 activation='silu',
                 last_activation=True,
                 dropout=0.1,
                 debug=False):
        super(SpectralRegressor, self).__init__()
        '''
        A wrapper for both SpectralConv1d and SpectralConv2d
        Ref: Li et 2020 FNO paper
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        A new implementation incoporating all spacial-based FNO
        in_dim: input dimension, (either n_hidden or spacial dim)
        n_hidden: number of hidden features out from attention to the fourier conv
        '''
        if spacial_dim == 2:  # 2d, function + (x,y)
            spectral_conv = SpectralConvWithFC
        elif spacial_dim == 1:  # 1d, function + x
            raise NotImplementedError("3D not implemented.")
        activation = default(activation, 'silu')
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        dropout = default(dropout, 0.1)
        self.spacial_fc = spacial_fc  # False in Transformer
        if self.spacial_fc:
            self.fc = nn.Linear(in_dim + spacial_dim, n_hidden)
        self.spectral_conv = nn.ModuleList([spectral_conv(in_channels=n_hidden,
                                                          out_channels=freq_dim,
                                                          modes1=modes1,
                                                          modes2=modes2,
                                                          n_grid=n_grid,
                                                          dropout=dropout,
                                                          activation=activation,
                                                          return_freq=return_freq,
                                                          debug=debug)])
        for _ in range(num_spectral_layers - 1):
            self.spectral_conv.append(spectral_conv(in_channels=freq_dim,
                                                    out_channels=freq_dim,
                                                    modes1=modes1,
                                                    modes2=modes2,
                                                    n_grid=n_grid,
                                                    dropout=dropout,
                                                    activation=activation,
                                                    return_freq=return_freq,
                                                    debug=debug))
        if not last_activation:
            self.spectral_conv[-1].activation = Identity()

        self.n_grid = n_grid  # dummy for debug
        self.dim_feedforward = default(dim_feedforward, 2*spacial_dim*freq_dim)
        self.regressor = nn.Sequential(
            nn.Linear(freq_dim, self.dim_feedforward),
            self.activation,
            nn.Linear(self.dim_feedforward, out_dim),
        )
        self.normalizer = normalizer
        self.return_freq = return_freq
        self.return_latent = return_latent
        self.debug = debug

    def forward(self, x, edge=None, pos=None, grid=None):
        '''
        2D:
            Input: (-1, n, n, in_features)
            Output: (-1, n, n, n_targets)
        1D:
            Input: (-1, n, in_features)
            Output: (-1, n, n_targets)
        '''
        x_latent = []
        x_fts = []

        if self.spacial_fc:
            x = torch.cat([x, grid], dim=-1)
            x = self.fc(x)
        for layer in self.spectral_conv:
            if self.return_freq:
                x, x_ft = layer(x)
                x_fts.append(x_ft.contiguous())
            else:
                x = layer(x)

            if self.return_latent:
                x_latent.append(x.contiguous())

        x = self.regressor(x)

        if self.normalizer:
            x = self.normalizer.inverse_transform(x)
        if self.return_freq or self.return_latent:
            return x, dict(preds_freq=x_fts, preds_latent=x_latent)
        else:
            return x

    
class FourierLayer2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FourierLayer2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.spec_conv = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, norm='ortho')
        self.norm_conv1d = nn.Conv1d(self.width, self.width, 1)

    def forward(self, x):
        batch_size, dim, dom_size1, dom_size2 = x.shape
        #x1 = self.spec_conv(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # use v2
        x1 = self.spec_conv(x)  # use v1
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


class RNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, recurrent_index, layer_num=3, pad_amount=None, pad_dim='1'):
        """
            `pad_dim` can be '1', '2', or 'both', and this decides which of the two space dimensions to pad
            `pad_amount` is a tuple that determines how much to pad each dimension by, if `pad_dim`
            specifies that dimension should be padded.
        """
        super(RNO2d, self).__init__()
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
                                           out_dim=self.out_dim, modes1=modes1, modes2=modes2, activation='relu', dropout=0.3) # NOTE: CHANGED `modes1`
    
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