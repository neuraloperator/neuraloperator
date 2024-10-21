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
        num_dim : int
            number of dimensions in the input domain
        kernel_size : int
            odd kernel size used for convolutional finite difference stencil
        groups : int
            splitting number of channels
        padding : literal {'periodic', 'replicate', 'reflect', 'zeros'}
            mode of padding to use on input. 
            See `torch.nn.functional.padding`. 
        implementation : literal {'subtract_middle', 'subtract_all'}
            for kernel c,

            * 'subtract_middle' computes 1/h \cdot (c * f - f(middle) \cdot \sum_{c_i})
            
            * 'subtract_all' computes 1/h \cdot (c_i - mean(c)) * f

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
            num_dim, 
            kernel_size=3, 
            groups=1, 
            padding='periodic', 
            implementation='subtract_middle'):
        super().__init__()
        conv_module = getattr(nn, f"Conv{num_dim}d")
        self.F_conv_module = getattr(F, f"conv{num_dim}d")
        self.conv_function = getattr(F, f"conv{num_dim}d")

        assert kernel_size % 2 == 1, "Kernel size should be odd"
        self.kernel_size = kernel_size

        self.in_channels = in_channels
        self.groups = groups
        self.num_dim = num_dim

        if padding == 'periodic':
            self.padding_mode = 'circular'
        elif padding == 'replicate':
            self.padding_mode = 'replicate'
        elif padding == 'reflect':
            self.padding_mode = 'reflect'
        elif padding == 'zeros':
            if implementation == 'subtract_middle':
                self.padding_mode = 'zeros'
            elif implementation == 'subtract_all':
                self.padding_mode = 'constant'
        else:
            raise NotImplementedError("Desired padding mode is not currently supported")
        self.pad_size = kernel_size // 2

        self.implementation = implementation
        if implementation == 'subtract_all':
            self.conv_kernel = torch.nn.parameter.Parameter(torch.randn(
                                        out_channels, in_channels // groups, kernel_size, kernel_size))
            self.weight = self.conv_kernel
        elif implementation == 'subtract_middle':
            self.conv = conv_module(in_channels, out_channels, kernel_size=kernel_size, 
                            padding='same', padding_mode=self.padding_mode,
                            bias=False, groups=groups)
            self.weight = self.conv.weight
        else:
            raise NotImplementedError("Desired implementation is not currently supported")

    def _forward_subtract_middle(self, x, grid_width):
        conv = self.conv(x)
        conv_sum = torch.sum(self.conv.weight, dim=tuple([i for i in range(2, 2 + self.num_dim)]), keepdim=True)
        conv_sum = self.conv_function(x, conv_sum, groups=self.groups)
        return (conv - conv_sum) / grid_width
    def _forward_subtract_all(self, x, grid_width):
        x_pad = F.pad(x, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), self.padding_mode)
        conv_mean = torch.mean(self.conv_kernel, dim=tuple([i for i in range(2, 2 + self.num_dim)]), keepdim=True)
        conv_mean = conv_mean.repeat([1 for _ in range(len(conv_mean.shape) - self.num_dim)] + [self.kernel_size for _ in range(self.num_dim)])
        conv_x = self.F_conv_module(input=x_pad, weight=(self.conv_kernel - conv_mean), padding='valid', groups=self.groups)
        return conv_x / grid_width

    def forward(self, x, grid_width):
        if self.implementation == 'subtract_middle':
            return self._forward_subtract_middle(x, grid_width)
        elif self.implementation == 'subtract_all':
            return self._forward_subtract_all(x, grid_width)
        else:
            raise RuntimeError