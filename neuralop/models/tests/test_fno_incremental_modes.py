"""
Training just one spectral layer to see if the Incremental Algorithm works - Author Robert Joseph
"""
import torch
import torch.nn as nn
from neuralop.layers.spectral_convolution import SpectralConv

device = 'cpu'

class Optimizer(torch.optim.SGD):
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # return the last gradient as this corresponds to the frequency
                # layer
                d_p = p.grad.data

                p.data.add_(d_p, alpha=-group['lr'])

        return loss, d_p

# %%
def test_grad_incremental():
    """_summary_
    Check if the gradients of the non used frequency modes are zero
    Check if the gradients of the used frequency modes are non zero
    We only check the corresponding dimensions of the gradients
    """
    # set up model
    modes = (6, 9, 12, 14)
    incremental_modes = (2, 3, 4, 5)
    index = list(incremental_modes)
    for dim in [1, 2, 3, 4]:

        print("Dimension of spectral conv: ", dim)
        conv = SpectralConv(
            2, 2, modes[:dim], n_layers=1, bias=False)

        original_weights = conv.weight[0].to_tensor().clone()

        x = torch.randn(2, 2, *(6, ) * dim, requires_grad=True)
        y = torch.randn(2, 2, *(6, ) * dim, requires_grad=True)

        # define a loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = Optimizer(conv.parameters(), lr=0.01)
        # run the input data through the model and update the weights

        # Dynamically reduce the number of modes in Fourier space
        conv.incremental_n_modes = incremental_modes[:dim]
        for i in range(5):
            res = conv(x)
            loss = criterion(res, y)
            optimizer.zero_grad()
            loss.backward()
            loss, grad = optimizer.step()

        # check the gradients of the non used incremental modes are zero
        assert torch.allclose(
            grad[:, :, index[dim - 1]:], torch.zeros_like(grad[:, :, index[dim - 1]:]))

        # check the gradients of the used incremental modes are non zero
        assert not torch.allclose(
            grad[:, :, :index[dim - 1]], torch.zeros_like(grad[:, :, :index[dim - 1]]))

        new_weights = conv.weight[0].to_tensor().clone()

        # same shape (no increase in dimensions)
        assert new_weights.shape == original_weights.shape

        print("Passed test for dimension: ", dim, "\n")

# Test method
test_grad_incremental()
