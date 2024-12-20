import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from ..layers.channel_mlp import ChannelMLP
from ..layers.spectral_convolution import SpectralConv
from ..layers.skip_connections import skip_connection
from ..layers.padding import DomainPadding
from ..layers.coda_blocks import CODABlocks
from ..layers.resample import resample
from ..layers.embeddings import GridEmbedding2D, GridEmbeddingND

class CODANO(nn.Module):
    """
    Parameters
    ---
    output_variable_codimension : int
        number of output channels corresponding to each variable
    hidden_variable_codimension : int
        number of hidden channels corresponding to each variable
    lifting_variable_codimension : int
        number of hidden channels in lifting/projection module for each variable
    use_positional_encoding : bool
        whether to use varibale specific positional encoding
    positional_encoding_dim : int
        dimension (number of channels) of positional encoding
    static_channel_dim : int
        number of channels in static channel. These channels contains additional static information of the PDE such as boundary conditions.
    positional_encoding_modes : list
        number of Fourier modes used in positional encoding
    n_variables : int
        number of variables
    use_horizontal_skip_connection : bool
        whether to use horizontal skip connections
    horizontal_skips_map : dict
                    a map {...., b: a, ....} denoting horizontal skip connection from a-th layer to
                    b-th layer.
                    Example: For a 5 layer UNO architecture, the skip connections can be
                    horizontal_skips_map ={4:0,3:1}
    n_layers : int
        number of layers
    n_modes : list
        Number of Fourier Modes to use in integral operations in the CoDA-NO block (along each dimension).
        Example: For a five layer CODANO with 2D input the n_modes can be: [[16,16],[16,16],[16,16],[16,16],[16,16]]
    scalings : list
        the output scaling factor for each layer along each dimension. 
        For example, for a 2D input, the scalings can be [[1,1],[0.5,0.5],[1,1], [2,2], [1,1]]
    n_heads : list
        number of attention heads for each layer
    attention_scalings : list
        scaling factor in attention mechanism. For example, for a 5 layer CoDA-NO, the attention_scalings can be [0.5, 0.5, 0.5, 0.5, 0.5].
    conv_module : nn.Module
        convolution module to use in the CoDA-NO block
    nonlinear_attention : bool
        whether to use non-linear attention mechanism
    non_linearity : nn.Module
        non-linearity to use in the CoDA-NO block
    attention_token_dim : int
        number of channels for each token for attention mechanism
    per_channel_attention : bool
        whether to use per channel attention mechanism
    layer_kwargs : dict
        additional arguments for the CODABlocks
    projection : bool
        whether to use projection module
    lifting : bool
        whether to use lifting module
    domain_padding : float
        padding factor for domain padding
    domain_padding_mode : str
        domain padding mode (one-sided or two-sided)
    enable_cls_token : bool
        whether to use CLS token
    """
    def __init__(
        self,
        output_variable_codimension=1,
        hidden_variable_codimension=32,
        lifting_variable_codimension=64,
        use_positional_encoding=False,
        positional_encoding_dim=0,
        static_channel_dim=0,
        positional_encoding_modes=None,
        n_variables=None,        
        use_horizontal_skip_connection=False,
        horizontal_skips_map=None,
        n_layers=4,
        n_modes=None,
        scalings=None,
        n_heads=None,
        attention_scalings=None,
        conv_module=SpectralConv,
        nonlinear_attention=False,
        non_linearity=F.gelu,
        attention_token_dim=1,
        per_channel_attention=True,
        layer_kwargs={},
        projection=True,
        lifting=True,
        domain_padding=0.5,
        domain_padding_mode='one-sided',
        enable_cls_token=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        assert len(n_modes) == n_layers, "number of modes for all layers are not given"
        assert len(n_heads) == n_layers, "number of Attention head for all layers are not given"
        assert len(scalings) == n_layers, "scaling for all layers are not given"
        assert len(attention_scalings) == n_layers, "attention scaling for all layers are not given"
        if use_positional_encoding:
            assert positional_encoding_dim > 0, "positional encoding dim is not given"
            assert positional_encoding_modes is not None, "positional encoding modes are not given"
        else:
            positional_encoding_dim = 0

        input_variable_codimension=1 # each channel is a variable

        self.n_dim = len(n_modes[0])
        self.input_variable_codimension = input_variable_codimension
        if hidden_variable_codimension is None:
            hidden_variable_codimension = input_variable_codimension
        if lifting_variable_codimension is None:
            lifting_variable_codimension = input_variable_codimension
        if output_variable_codimension is None:
            output_variable_codimension = input_variable_codimension

        self.hidden_variable_codimension = hidden_variable_codimension
        self.n_modes = n_modes
        self.scalings = scalings
        self.non_linearity = non_linearity
        self.n_heads = n_heads
        self.lifting = lifting
        self.projection = projection
        self.enable_cls_token = enable_cls_token
        self.positional_encoding_dim = positional_encoding_dim
        self.n_variables = n_variables
        self.attention_scalings = attention_scalings
        self.positional_encoding_modes = positional_encoding_modes
        self.static_channel_dim = static_channel_dim
        self.layer_kwargs = layer_kwargs
        self.use_positional_encoding = use_positional_encoding
        self.use_horizontal_skip_connection = use_horizontal_skip_connection
        self.horizontal_skips_map = horizontal_skips_map
        self.output_variable_codimension = output_variable_codimension

        if self.positional_encoding_modes is not None:
            self.positional_encoding_modes[-1] = self.positional_encoding_modes[-1]//2

        # calculating scaling
        if self.scalings is not None:
            self.end_to_end_scaling = [1] * len(self.scalings[0])
            # multiplying scaling factors
            for k in self.scalings:
                self.end_to_end_scaling = [
                    i * j for (i, j) in zip(self.end_to_end_scaling, k)
                ]
        else:
            self.end_to_end_scaling = [1] * self.n_dim

        # Setting up domain padding for encoder and reconstructor
        if domain_padding is not None and domain_padding > 0:
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                resolution_scaling_factor=self.end_to_end_scaling,
            )
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode

        extended_variable_codimemsion = input_variable_codimension + static_channel_dim + positional_encoding_dim
        self.extended_variable_codimemsion = extended_variable_codimemsion
        if self.lifting:
            self.lifting = ChannelMLP(
            in_channels=extended_variable_codimemsion,
            out_channels=hidden_variable_codimension,
            hidden_channels=lifting_variable_codimension,
            n_layers=2,
            n_dim=self.n_dim,
            )
        else:
            self.lifting = None

        self.base = nn.ModuleList([])
        
        for i in range(self.n_layers):
            self.base.append(
                CODABlocks(
                    n_modes=self.n_modes[i],
                    n_heads=self.n_heads[i],
                    scale=self.attention_scalings[i],
                    token_codimension=attention_token_dim,
                    per_channel_attention=per_channel_attention,
                    nonlinear_attention = nonlinear_attention,
                    resolution_scaling_factor=self.scalings[i],
                    conv_module=conv_module,
                    non_linearity=self.non_linearity,
                    **self.layer_kwargs,
                )
            )

        if self.use_horizontal_skip_connection:
            self.skip_map_module = nn.ModuleDict()
            for k in self.horizontal_skips_map.keys():
                self.skip_map_module[str(k)] = ChannelMLP(
                    in_channels=2*hidden_variable_codimension,
                    out_channels=hidden_variable_codimension,
                    hidden_channels=None,
                    n_layers=1,
                    non_linearity=nn.Identity(),
                    n_dim=self.n_dim,
                )

        if self.projection:
            self.projection = ChannelMLP(
                in_channels=hidden_variable_codimension,
                out_channels=output_variable_codimension,
                hidden_channels=lifting_variable_codimension,
                n_layers=2,
                n_dim=self.n_dim)
        else:
            self.projection = None

        if enable_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1,
            hidden_variable_codimension, **self.positional_encoding_modes, dtype=torch.cfloat))
        
        if use_positional_encoding:
            self.positional_encoding = nn.ParameterList()
            for i in range(self.n_variables):
                self.positional_encoding.append(
                    nn.Parameter(
                        torch.randn(1,
                            positional_encoding_dim,
                            *self.positional_encoding_modes,
                            dtype=torch.cfloat,
                        )
                    )
                )
    def _extend_positional_encoding(self, num_variables):
        """
        Add positional encoding for new variables.

        Parameters
        ----------
        num_variables : int
            number of variables to add
        """
        for i in range(num_variables):
            self.positional_encoding.append(
                nn.Parameter(
                    torch.randn(1,
                        self.positional_encoding_dim,
                        *self.positional_encoding_modes,
                        dtype=torch.cfloat,
                    )
                )
            )
        self.n_variables += num_variables

    def _get_positional_encoding(self, x):
        encoding_list = []
        for i in range(self.n_variables):
            encoding_list.append(torch.fft.irfftn(self.positional_encoding[i],
                                                s=x.shape[-self.n_dim:]))
        
        return torch.stack(encoding_list, dim=1)
    
    def _get_cls_token(self, x):
        cls_token = torch.fft.irfftn(self.cls_token, s=x.shape[-self.n_dim:])
        repeat_shape = [1 for _ in x.shape]
        repeat_shape[0] = x.shape[0]
        cls_token = cls_token.repeat(*repeat_shape)
        return cls_token
    
    def _extend_variables(self, x, static_channel):
        x = x.unsqueeze(2)
        if static_channel is not None:
            repeat_shape = [1 for _ in x.shape]
            repeat_shape[1] = x.shape[1]
            static_channel = static_channel.unsqueeze(1).repeat(*repeat_shape)
            x = torch.cat([x, static_channel], dim=2)
        if self.use_positional_encoding:
            positional_encoding = self._get_positional_encoding(x)
            repeat_shape = [1 for _ in x.shape]
            repeat_shape[0] = x.shape[0]
            x = torch.cat([x, positional_encoding.repeat(*repeat_shape)], dim=2)
        return x

    def forward(self, x: torch.Tensor, static_channel=None):
        '''
        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, channel, H, W, ...)
        static_channel : torch.Tensor
            static channel tensor of shape (batch_size, static_channel_dim, H, W, ...)
        '''
        batch, n_variables, *spatial_shape = x.shape

        if self.static_channel_dim > 0 and static_channel is None and static_channel.shape[1] != self.static_channel_dim:
            raise ValueError("static channel dimension is not correct")
        if self.use_positional_encoding and self.n_variables is None:
            raise ValueError("number of variables is not given")
        if self.use_positional_encoding:
            assert x.shape[1]//self.n_variables == self.input_variable_codimension, "input variable codimension is not correct"
        
        x = self._extend_variables(x, static_channel) # (batch_size, n_variables, extended_codim, H, W, ...)

        if self.lifting:
            x = x.reshape(batch*n_variables, self.extended_variable_codimemsion, *spatial_shape)
            x = self.lifting(x)
            x = x.reshape(batch, n_variables*self.hidden_variable_codimension, *spatial_shape)

        if self.enable_cls_token:
            cls_token = self._get_cls_token(x)
            x = torch.cat(
                [
                    cls_token,
                    x,
                ],
                dim=1,
            )

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        output_shape = [
            int(round(i * j))
            for (i, j) in zip(x.shape[-self.n_dim :], self.end_to_end_scaling)
        ]

        skip_outputs = {}
        for layer_idx in range(self.n_layers):

            if self.horizontal_skips_map is not None and layer_idx in self.horizontal_skips_map.keys():
                skip_val = skip_outputs[self.horizontal_skips_map[layer_idx]]
                resolution_scaling_factors = [
                    m / n for (m, n) in zip(x.shape, skip_val.shape)
                ]
                resolution_scaling_factors = resolution_scaling_factors[-1 * self.n_dim :]
                t = resample(
                    skip_val, resolution_scaling_factors, list(range(-self.n_dim, 0)), output_shape=x.shape[-self.n_dim:]
                )
                x = x.reshape(batch*n_variables, self.hidden_variable_codimension, *x.shape[-self.n_dim:])
                t = t.reshape(batch*n_variables, self.hidden_variable_codimension, *t.shape[-self.n_dim:])
                x = torch.cat([x, t], dim=1)
                x = self.skip_map_module[str(layer_idx)](x)
                x = x.reshape(batch, n_variables*self.hidden_variable_codimension, *x.shape[-self.n_dim:])

            if layer_idx == self.n_layers - 1:
                cur_output_shape = output_shape
            else:
                cur_output_shape = None

            x = self.base[layer_idx](x, output_shape=cur_output_shape)

            if self.horizontal_skips_map is not None and layer_idx in self.horizontal_skips_map.values():
                skip_outputs[layer_idx] = x.clone()

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        if self.projection:
            x = x.reshape(batch*n_variables, self.hidden_variable_codimension, *x.shape[-self.n_dim:])
            x = self.projection(x)
            x = x.reshape(batch, n_variables*self.output_variable_codimension, *x.shape[-self.n_dim:])

        if self.enable_cls_token:
            x = x[:, self.output_variable_codimension:, ...]
        return x