"""
Authors: Miguel Liu-Schiaffini (mliuschi@caltech.edu) and Zelin Zhao (sjtuytc@gmail.com)

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

from spectral_convolution import FactorizedSpectralConv2d

torch.manual_seed(0)
np.random.seed(0)


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


class FourierLayer2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FourierLayer2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        #self.conv = SpectralConv2d(self.width, self.width, self.modes1 // 2, self.modes2 // 2)
        #self.conv = FactorizedSpectralConv2d(self.width, self.width, (self.modes1, self.modes2), n_layers=1, fft_norm='ortho', implementation='factorized')
        #self.conv = FactorizedSpectralConv2d(self.width, self.width, (self.modes1, self.modes2), n_layers=1, fft_norm='forward', implementation='factorized')
        #self.conv = FactorizedSpectralConv2d(self.width, self.width, (self.modes1, self.modes2), n_layers=1, fft_norm='forward', factorization=None)
        self.conv = FactorizedSpectralConv2d(self.width, self.width, (self.modes1, self.modes2), n_layers=1, fft_norm='forward', factorization=None, separable=False)
        self.w = nn.Conv1d(self.width, self.width, 1)

    def forward(self, x):
        batch_size, dim, dom_size1, dom_size2 = x.shape
        
        x1 = self.conv(x)
        x2 = self.w(x.reshape((batch_size, dim, dom_size1 * dom_size2))).view(batch_size, self.width, dom_size1, dom_size2)

        return x1 + x2

class RNO_cell(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(RNO_cell, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.f1 = FourierLayer2d(self.modes1, self.modes2, self.width)
        self.f2 = FourierLayer2d(self.modes1, self.modes2, self.width)
        self.f3 = FourierLayer2d(self.modes1, self.modes2, self.width)
        self.f4 = FourierLayer2d(self.modes1, self.modes2, self.width)
        self.f5 = FourierLayer2d(self.modes1, self.modes2, self.width)
        self.f6 = FourierLayer2d(self.modes1, self.modes2, self.width)

        self.b1 = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.))) # constant bias terms
        self.b2 = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.)))
        self.b3 = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.)))
    
    def forward(self, x, h):
        z = torch.sigmoid(self.f1(x) + self.f2(h) + self.b1)
        r = torch.sigmoid(self.f3(x) + self.f4(h) + self.b2)
        h_hat = F.selu(self.f5(x) + self.f6(r * h) + self.b3) # selu for regression problem

        h_next = (1. - z) * h + z * h_hat

        return h_next

class RNO_layer(nn.Module):
    def __init__(self, modes1, modes2, width, return_sequences=False):
        super(RNO_layer, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.return_sequences = return_sequences

        self.cell = RNO_cell(modes1, modes2, width)
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


class RNO_2D(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, modes1, modes2, width, pad_amount=None, pad_dim='1', use_grid=True):
        """
            `pad_dim` can be '1', '2', or 'both', and this decides which of the two space dimensions to pad
            `pad_amount` is a tuple that determines how much to pad each dimension by, if `pad_dim`
            specifies that dimension should be padded.
        """
        super(RNO_2D, self).__init__()

        self.modes1 = modes1
        self.modes1 = modes2
        self.num_layers = num_layers
        self.width = width
        self.pad_amount = pad_amount # pads dom_size1 dimension
        self.pad_dim = pad_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_grid = use_grid

        if use_grid:
            self.lifting = nn.Linear(in_dim + 2, self.width)
        else:
            self.lifting = nn.Linear(in_dim, self.width)

        module_list = [RNO_layer(modes1, modes2, width, return_sequences=True)
                                     for _ in range(num_layers - 1)]
        module_list.append(RNO_layer(modes1, modes2, width, return_sequences=False))
        self.layers = nn.ModuleList(module_list)

        self.projection = nn.Linear(self.width, out_dim)
    
    def forward(self, x, init_hidden_states=None): # h must be padded if using padding
        batch_size, timesteps, dom_size1, dom_size2, dim = x.shape

        if init_hidden_states is None:
            init_hidden_states = [None] * self.num_layers

        if self.use_grid:
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=-1)
        
        x = self.lifting(x)

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
        for i in range(self.num_layers):
            pred_x = self.layers[i](x, init_hidden_states[i])
            # using residual predictions
            if i < self.num_layers - 1:
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
        pred = self.projection(h)

        return pred, final_hidden_states

    def predict(self, x, num_steps): # num_steps is the number of steps ahead to predict
        output = []
        states = [None] * self.num_layers
        
        for i in range(num_steps):
            pred, states = self.forward(x, states)
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
        # Credit: Vadim Smolyakov on PyTorch forum
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return int(sum([np.prod(p.size()) for p in model_parameters]))