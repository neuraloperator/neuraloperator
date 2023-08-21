"""
Authors: Miguel Liu-Schiaffini (mliuschi@caltech.edu) and Zelin Zhao (sjtuytc@gmail.com)

This file implements a recurrent neural operator (RNO) based on the gated recurrent unit (GRU)
and Fourier neural operator (FNO) architectures.

In particular, the RNO has an identical architecture to the finite-dimensional GRU, 
with the exception that linear matrix-vector multiplications are replaced by linear 
Fourier layers (see Li et al., 2021), and for regression problems, the output nonlinearity
is replaced with a SELU activation.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from spectral_convolution import FactorizedSpectralConv1d, FactorizedSpectralConv2d, FactorizedSpectralConv3d
from padding import DomainPadding

torch.manual_seed(0)
np.random.seed(0)

class FourierLayer(nn.Module):
    def __init__(self, n_modes, width, fft_norm='ortho', factorization=None, separable=False):
        """
            n_modes : tuple of modes
            width : int of width of spectral convolution
        """
        super(FourierLayer, self).__init__()

        self.width = width

        if len(n_modes) == 1:
            spec_conv = FactorizedSpectralConv1d
        elif len(n_modes) == 2:
            spec_conv = FactorizedSpectralConv2d
        elif len(n_modes) == 3:
            spec_conv = FactorizedSpectralConv3d
        else:
            raise NotImplementedError

        self.conv = spec_conv(width, width, n_modes, n_layers=1, fft_norm=fft_norm, factorization=factorization, separable=separable)
        self.w = nn.Conv1d(self.width, self.width, 1)

    def forward(self, x):
        batch_size, dim, dom_size1, dom_size2 = x.shape
        
        x1 = self.conv(x)
        x2 = self.w(x.reshape((batch_size, dim, dom_size1 * dom_size2))).view(batch_size, self.width, dom_size1, dom_size2)

        return x1 + x2

class RNO_cell(nn.Module):
    def __init__(self, n_modes, width, fft_norm='ortho', factorization=None, separable=False):
        super(RNO_cell, self).__init__()

        self.width = width

        self.f1 = FourierLayer(n_modes, self.width, fft_norm=fft_norm, factorization=factorization, separable=separable)
        self.f2 = FourierLayer(n_modes, self.width, fft_norm=fft_norm, factorization=factorization, separable=separable)
        self.f3 = FourierLayer(n_modes, self.width, fft_norm=fft_norm, factorization=factorization, separable=separable)
        self.f4 = FourierLayer(n_modes, self.width, fft_norm=fft_norm, factorization=factorization, separable=separable)
        self.f5 = FourierLayer(n_modes, self.width, fft_norm=fft_norm, factorization=factorization, separable=separable)
        self.f6 = FourierLayer(n_modes, self.width, fft_norm=fft_norm, factorization=factorization, separable=separable)

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
    def __init__(self, n_modes, width, return_sequences=False, fft_norm='ortho', factorization=None, separable=False):
        super(RNO_layer, self).__init__()

        self.width = width
        self.return_sequences = return_sequences

        self.cell = RNO_cell(n_modes, width, fft_norm=fft_norm, factorization=factorization, separable=separable)
        self.bias_h = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.)))

    def forward(self, x, h=None):
        batch_size, timesteps, dim = x.shape[:3]
        dom_sizes = x.shape[3:]

        if h is None:
            h = torch.zeros((batch_size, self.width, *dom_sizes)).to(x.device)
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


class RNO(nn.Module):
    def __init__(self, in_dim, out_dim, 
                num_layers, 
                n_modes, 
                width, 
                residual=False, 
                domain_padding=None, 
                domain_padding_mode='one-sided', 
                use_grid=True, 
                output_scaling_factor=None,
                fft_norm='ortho', 
                factorization=None, 
                separable=False
                ):
        """
            General N-D RNO implementation:
                `domain_padding` : float list, optional
                    If not None, percentage of padding to use, by default None
                    Must be a list
                `domain_padding_mode` : {'symmetric', 'one-sided'}, optional
                    How to perform domain padding, by default 'one-sided'
                `residual` : whether to use residual connections in the hidden layers
                `output_scaling_factor` : scaling factor of output resolution
        """
        super(RNO, self).__init__()

        self.n_modes = n_modes
        self.n_dims = len(n_modes)
        self.num_layers = num_layers
        self.width = width

        if domain_padding is not None and ((isinstance(domain_padding, list) and sum(domain_padding) > 0)):
                domain_padding = [0,] + domain_padding # avoid padding channel dimension
                self.domain_padding = DomainPadding(domain_padding=domain_padding, padding_mode=domain_padding_mode, output_scaling_factor=output_scaling_factor)
        else:
            self.domain_padding = None

        self.domain_padding_mode = domain_padding_mode

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_grid = use_grid
        self.residual = residual

        if use_grid:
            self.lifting = nn.Linear(in_dim + len(n_modes), self.width)
        else:
            self.lifting = nn.Linear(in_dim, self.width)

        module_list = [RNO_layer(n_modes, width, return_sequences=True, fft_norm=fft_norm, factorization=factorization, separable=separable)
                                     for _ in range(num_layers - 1)]
        module_list.append(RNO_layer(n_modes, width, return_sequences=False, fft_norm=fft_norm, factorization=factorization, separable=separable))
        self.layers = nn.ModuleList(module_list)

        self.projection = nn.Linear(self.width, out_dim)
    
    def forward(self, x, init_hidden_states=None): # h must be padded if using padding
        batch_size, timesteps = x.shape[:2]
        dim = x.shape[-1]
        dom_sizes = x.shape[2 : 2 + self.n_dims]
        x_size = len(x.shape)

        if init_hidden_states is None:
            init_hidden_states = [None] * self.num_layers

        if self.use_grid:
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=-1)
        
        x = self.lifting(x)

        x = torch.movedim(x, x_size - 1, 2) # new shape: (batch, timesteps, dim, dom_size1, dom_size2, ..., dom_sizen)

        x = self.domain_padding.pad(x)

        final_hidden_states = []
        for i in range(self.num_layers):
            pred_x = self.layers[i](x, init_hidden_states[i])
            if i < self.num_layers - 1:
                if self.residual:
                    x = x + pred_x
                else:
                    x = pred_x
                final_hidden_states.append(x[:, -1])
            else:
                x = pred_x
                final_hidden_states.append(x)
        h = final_hidden_states[-1]

        h = h.unsqueeze(1) # add dim for padding compatibility
        #print(self.domain_padding._unpad_indices[f"{list(h.shape[2:])}"])
        h = self.domain_padding.unpad(h)
        h = h[:,0] # remove extraneous dim
        #print(h.shape)

        h = torch.movedim(h, 1, x_size - 2)
        #print(h.shape)

        #assert False

        pred = self.projection(h)

        return pred, final_hidden_states

    def predict(self, x, num_steps): # num_steps is the number of steps ahead to predict
        output = []
        states = [None] * self.num_layers
        
        for i in range(num_steps):
            pred, states = self.forward(x, states)
            output.append(pred)
            x = pred.reshape((pred.shape[0], 1, *pred.shape[1:]))

        return torch.stack(output, dim=1)

    def get_grid(self, shape, device):
        batch_size, steps = shape[:2]
        dim = shape[-1]
        dom_sizes = shape[2 : 2 + self.n_dims]

        grids = []
        for i in range(len(dom_sizes)):
            size = dom_sizes[i]
            grid = torch.tensor(np.linspace(0, 1, size), dtype=torch.float)
            grid = grid.reshape([1 for j in range(2 + i)] + [size] + [1 for j in range(len(shape) - 3 - i)]).repeat([batchsize, steps] + shape[2 : 2 + i] + [1] + shape[3 + i : -1] + [1])
            grids.append(grid)

        return torch.cat(grids, dim=-1).to(device)

    def count_params(self):
        # Credit: Vadim Smolyakov on PyTorch forum
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return int(sum([np.prod(p.size()) for p in model_parameters]))