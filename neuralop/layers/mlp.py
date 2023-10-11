import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """A Multi-Layer Perceptron, with arbitrary number of layers

    Parameters
    ----------
    in_channels : int
    out_channels : int, default is None
        if None, same is in_channels
    hidden_channels : int, default is None
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : default is F.gelu
    dropout : float, default is 0
        if > 0, dropout probability
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
        **kwargs,
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

        # Decide if we're using Conv layers or Linear layers based on n_dim
        if n_dim < 4:
            Conv = getattr(nn, f"Conv{n_dim}d")
            self.fcs = nn.ModuleList()
            for i in range(n_layers):
                if i == 0 and i == (n_layers - 1):
                    self.fcs.append(Conv(self.in_channels, self.out_channels, 1))
                elif i == 0:
                    self.fcs.append(Conv(self.in_channels, self.hidden_channels, 1))
                elif i == (n_layers - 1):
                    self.fcs.append(Conv(self.hidden_channels, self.out_channels, 1))
                else:
                    self.fcs.append(Conv(self.hidden_channels, self.hidden_channels, 1))
        else:
            if n_layers == 1:
                layers = [self.in_channels, self.out_channels]
            else:
                layers = [self.in_channels, self.hidden_channels, self.out_channels]
            
            self.fcs = nn.ModuleList()
            for j in range(n_layers):
                self.fcs.append(nn.Linear(layers[j], layers[j + 1]))

    def forward(self, x):
        size = list(x.shape)
        x = x.reshape(size[0], size[1], -1).permute(0, -1, 1)
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)  # Correctly use the dropout from the ModuleList
        size[1] = self.out_channels
        return x.permute(0, -1, 1).reshape(size)


# Reimplementation of the MLP class using Linear instead of Conv
class MLPLinear(torch.nn.Module):
    def __init__(self, layers, non_linearity=F.gelu, dropout=0.0):
        super().__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.fcs = nn.ModuleList()
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(self.n_layers)])
            if dropout > 0.0
            else None
        )

        for j in range(self.n_layers):
            self.fcs.append(nn.Linear(layers[j], layers[j + 1]))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout(x)

        return x