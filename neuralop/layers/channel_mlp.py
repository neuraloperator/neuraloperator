import torch
from torch import nn
import torch.nn.functional as F


class ChannelMLP(nn.Module):
    """Multi-layer perceptron applied channel-wise across spatial dimensions.

    ChannelMLP applies a series of 1D convolutions and nonlinearities to the channel
    dimension of input tensors, making it invariant to spatial resolution. This is
    particularly useful in neural operators where the spatial dimensions may vary
    but the channel processing should remain consistent.

    The implementation uses 1D convolutions with kernel size 1, which effectively
    performs linear transformations on the channel dimension while preserving
    spatial structure. This approach is more efficient than reshaping to 2D and
    using fully connected layers.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int, optional
        Number of output channels. If None, defaults to in_channels.
    hidden_channels : int, optional
        Number of hidden channels in intermediate layers. If None, defaults to in_channels.
    n_layers : int, optional
        Number of linear layers in the MLP, by default 2
    n_dim : int, optional
        Spatial dimension of input (unused but kept for compatibility), by default 2
    non_linearity : callable, optional
        Nonlinear activation function to apply between layers, by default F.gelu
    dropout : float, optional
        Dropout probability applied after each layer (except the last).
        If 0, no dropout is applied, by default 0.0
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        hidden_channels=None,
        n_layers=2,
        n_dim=2,
        non_linearity=F.gelu,
        dropout=0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
            if dropout > 0.0
            else None
        )

        # Build the MLP layers using 1D convolutions with kernel size 1
        # This effectively performs linear transformations on the channel dimension
        # while preserving spatial structure and being more efficient than FC layers
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                # Single layer: input -> output
                self.fcs.append(nn.Conv1d(self.in_channels, self.out_channels, 1))
            elif i == 0:
                # First layer: input -> hidden
                self.fcs.append(nn.Conv1d(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                # Last layer: hidden -> output
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.out_channels, 1))
            else:
                # Internal layers: hidden -> hidden
                self.fcs.append(
                    nn.Conv1d(self.hidden_channels, self.hidden_channels, 1)
                )

    def forward(self, x):
        """
        Forward pass through the channel MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, *spatial_dims)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_channels, *spatial_dims)
        """
        reshaped = False
        size = list(x.shape)

        # Handle high-dimensional inputs (4D+) by flattening spatial dimensions
        # This allows the 1D convolutions to process all spatial positions uniformly
        if x.ndim > 3:
            # Flatten spatial dimensions: (batch, channels, x1, x2, ...) -> (batch, channels, -1)
            # Use reshape() instead of view() to handle non-contiguous tensors
            x = x.reshape((*size[:2], -1))
            reshaped = True

        # Apply MLP layers with nonlinearity and dropout
        for i, fc in enumerate(self.fcs):
            x = fc(x)  # Linear transformation (1D conv with kernel size 1)
            if i < self.n_layers - 1:  # Apply nonlinearity to all layers except the last
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        # Restore original spatial dimensions if input was reshaped
        if reshaped:
            x = x.reshape((size[0], self.out_channels, *size[2:]))

        return x


class LinearChannelMLP(torch.nn.Module):
    """
    Multi-layer perceptron using fully connected layers for channel processing.

    This is an alternative implementation of ChannelMLP that uses standard Linear
    layers instead of 1D convolutions.

    Parameters
    ----------
    layers : list of int
        List defining the architecture: [in_channels, hidden1, hidden2, ..., out_channels]
        Must have at least 2 elements (input and output channels)
    non_linearity : callable, optional
        Nonlinear activation function to apply between layers, by default F.gelu
    dropout : float, optional
        Dropout probability applied after each layer (except the last).
        If 0, no dropout is applied, by default 0.0
    """

    def __init__(self, layers, non_linearity=F.gelu, dropout=0.0):
        super().__init__()

        self.n_layers = len(layers) - 1

        assert (
            self.n_layers >= 1
        ), "Error: trying to instantiate \
            a LinearChannelMLP with only one linear layer."

        self.fcs = nn.ModuleList()
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(self.n_layers)])
            if dropout > 0.0
            else None
        )

        # Build linear layers based on the provided architecture
        for j in range(self.n_layers):
            self.fcs.append(nn.Linear(layers[j], layers[j + 1]))

    def forward(self, x):
        """
        Forward pass through the linear channel MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels) or (batch*spatial, in_channels)
            Note: Input must be pre-reshaped to 2D format

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_channels) or (batch*spatial, out_channels)
            Note: Output needs to be reshaped back to spatial dimensions if needed
        """
        # Apply linear layers with nonlinearity and dropout
        for i, fc in enumerate(self.fcs):
            x = fc(x)  # Linear transformation
            if i < self.n_layers - 1:  # Apply nonlinearity to all layers except the last
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        return x
