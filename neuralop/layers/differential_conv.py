import torch
import torch.nn as nn
import torch.nn.functional as F

class FiniteDifferenceConvolution(nn.Module):
    """Finite Difference Convolution Layer introduced in [1]_.
        "Neural Operators with Localized Integral and Differential Kernels" (ICML 2024)
            https://arxiv.org/abs/2402.16845 

        Computes a finite difference convolution on a regular grid, 
        which converges to a directional derivative as the grid is refined.

        Parameters
        ----------
        in_channels : int
            number of in_channels
        out_channels : int
            number of out_channels
        n_dim : int
            number of dimensions in the input domain
        kernel_size : int
            odd kernel size used for convolutional finite difference stencil
        groups : int
            splitting number of channels
        padding : literal {'periodic', 'replicate', 'reflect', 'zeros'}
            mode of padding to use on input. 
            See `torch.nn.functional.padding`. 

        References
        ----------
        .. [1] : Liu-Schiaffini, M., et al. (2024). "Neural Operators with 
            Localized Integral and Differential Kernels". 
            ICML 2024, https://arxiv.org/abs/2402.16845. 

        """
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            n_dim, 
            kernel_size=3, 
            groups=1, 
            padding='periodic'):
        super().__init__()
        conv_module = getattr(nn, f"Conv{n_dim}d")
        self.F_conv_module = getattr(F, f"conv{n_dim}d")
        self.conv_function = getattr(F, f"conv{n_dim}d")

        assert kernel_size % 2 == 1, "Kernel size should be odd"
        self.kernel_size = kernel_size

        self.in_channels = in_channels
        self.groups = groups
        self.n_dim = n_dim

        if padding == 'periodic':
            self.padding_mode = 'circular'
        elif padding == 'replicate':
            self.padding_mode = 'replicate'
        elif padding == 'reflect':
            self.padding_mode = 'reflect'
        elif padding == 'zeros':
            self.padding_mode = 'zeros'
        else:
            raise NotImplementedError("Desired padding mode is not currently supported")
        self.pad_size = kernel_size // 2

        self.conv = conv_module(in_channels, out_channels, kernel_size=kernel_size, 
                                padding='same', padding_mode=self.padding_mode,
                                bias=False, groups=groups)
        self.weight = self.conv.weight

    def forward(self, x, grid_width):
        """FiniteDifferenceConvolution's forward pass. Alternatively,
        one could center the conv kernel by subtracting the mean pointwise
        in the kernel: ``conv(x, kernel - mean(kernel)) / grid_width``

        Parameters
        ----------
        x : torch.tensor
            input tensor, shape (batch, in_channels, d_1, d_2, ...d_n)
        grid_width : float
            discretization size of input grid
        """
        conv = self.conv(x)
        conv_sum = torch.sum(self.conv.weight, dim=tuple([i for i in range(2, 2 + self.n_dim)]), keepdim=True)
        conv_sum = self.conv_function(x, conv_sum, groups=self.groups)
        return (conv - conv_sum) / grid_width