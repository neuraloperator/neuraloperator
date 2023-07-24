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
    kwargs :  Dict[str, Any], optional
        Args to pass to convolution layers.
        Args include ``"device"``, ``"dtype"``.
    """

    def __init__(
            self,
            in_channels,
            out_channels=None,
            hidden_channels=None,
            n_layers=2,
            n_dim=2,
            non_linearity=F.gelu,
            dropout=0.,
            **kwargs
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = (in_channels
                             if out_channels is None
                             else out_channels)
        self.hidden_channels = (in_channels
                                if hidden_channels is None
                                else hidden_channels)
        self.non_linearity = non_linearity
        self.dropout = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(n_layers)]
        ) if dropout > 0. else None

        Conv = getattr(nn, f'Conv{n_dim}d')
        dtype = kwargs.get('dtype', None)
        device = kwargs.get('device', None)
        self.fcs = nn.ModuleList()
        if n_layers == 1:
            self.fcs.append(Conv(
                self.in_channels,
                self.out_channels,
                1,  # kernel_size
                dtype=dtype,
                device=device,
            ))
            return

        # First layer (of n>1 layers):
        self.fcs.append(Conv(
            self.in_channels,
            self.hidden_channels,
            1,  # kernel_size
            dtype=dtype,
            device=device,
        ))
        # Middle layers (this may be un-run if n_layers == 2):
        for i in range(1, n_layers - 1):
            self.fcs.append(Conv(
                self.hidden_channels,
                self.hidden_channels,
                1,  # kernel_size
                dtype=dtype,
                device=device,
            ))
        # Last layer:
        self.fcs.append(Conv(
            self.hidden_channels,
            self.out_channels,
            1,  # kernel_size
            dtype=dtype,
            device=device,
        ))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            # Don't apply non-linearity on the last layer:
            if i < (self.n_layers - 1):
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout(x)

        return x
