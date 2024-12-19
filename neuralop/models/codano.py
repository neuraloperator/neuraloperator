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

class CodANO(nn.Module):
    """
    Parameters
    ---
    
    """

    def __init__(
        self,
        input_variable_codimension=1,
        output_variable_codimension=1,
        hidden_variable_codimension=32,
        lifting_variable_codimension=64,
        use_positional_encoding=False,
        positional_encoding_dim=16,
        positional_encoding_modes=None,
        num_variables=None,        
        horizontal_skip_connection=False,
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
        integral_operator=None,
        layer_kwargs={},
        projection=True,
        lifting=True,
        domain_padding=0.5,
        domain_padding_mode='one-sided',
        n_variables=None,
        enable_cls_token=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        assert len(n_modes) == n_layers, "number of modes for all layers are not given"
        assert len(n_heads) == n_layers, "number of Attention head for all layers are not given"
        assert len(scalings) == n_layers, "scaling for all layers are not given"
        assert len(attention_scalings) == n_layers, "attention scaling for all layers are not given"

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
        self.integral_operator = integral_operator
        self.lifting = lifting
        self.projection = projection
        self.enable_cls_token = enable_cls_token
        self.positional_encoding_dim = positional_encoding_dim
        self.n_variables = n_variables
        self.positional_encoding_modes = positional_encoding_modes
        self.layer_kwargs = layer_kwargs

        self.positional_encoding_modes[-1] = self.positional_encoding_modes[-1]//2


        # calculating scaling
        if self.scalings is not None:
            self.end_to_end_scaling_factor = [1] * len(self.scalings[0])
            # multiplying scaling factors
            for k in self.scalings:
                self.end_to_end_scaling_factor = [
                    i * j for (i, j) in zip(self.end_to_end_scaling_factor, k)
                ]
        else:
            self.end_to_end_scaling = [1] * self.n_dim

        # Setting up domain padding for encoder and reconstructor
        if domain_padding is not None and domain_padding > 0:
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                output_scaling_factor=self.end_to_end_scaling,
            )
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode

        if self.lifting:
            self.lifting = ChannelMLP(
            in_channels=input_variable_codimension,
            out_channels=hidden_variable_codimension,
            hidden_channels=lifting_variable_codimension,
            n_layers=2,
            n_dim=self.n_dim,
            )
        else:
            self.lifting = None
        


        # cls_dimension = 1 if enable_cls_token else 0
        # self.codimension_size = hidden_variable_codimension * n_variables + cls_dimension

        self.base = nn.ModuleList([])
        
        for i in range(self.n_layers):
            self.base.append(
                CODABlocks(
                    n_modes=self.n_modes[i],
                    n_heads=self.n_heads[i],
                    scale=self.scalings[i],
                    token_codimension=attention_token_dim,
                    per_channel_attention=per_channel_attention,
                    nonlinear_attention = nonlinear_attention,
                    conv_module=conv_module,
                    non_linearity=self.non_linearity,
                    **self.layer_kwargs,
                )
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
    # def get_output_scaling_factor(self, initial_scale, scalings_per_layer):
    #     for k in scalings_per_layer:
    #         initial_scale = np.multiply(initial_scale, k)
    #     initial_scale = initial_scale.tolist()
    #     if len(initial_scale) == 1:
    #         initial_scale = initial_scale[0]
    #     return initial_scale

    # def get_device(self,):
    #     return self.cls_token.coefficients_r.device

    def forward(self, x: torch.Tensor):
        if self.lifting:
            x = self.lifting(x)

        if self.enable_cls_token:
            cls_token = self.cls_token(x).unsqueeze(0)
            repeat_shape = [1 for _ in x.shape]
            repeat_shape[0] = x.shape[0]
            x = torch.cat(
                [
                    cls_token.repeat(*repeat_shape),
                    x,
                ],
                dim=1,
            )

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        output_shape_en = [round(i * j) for (i,
                                             j) in zip(x.shape[-self.n_dim:],
                                                       self.end_to_end_scaling)]

        cur_output_shape = None
        for layer_idx in range(self.n_layers):
            if layer_idx == self.n_layers - 1:
                cur_output_shape = output_shape_en
            x = self.base[layer_idx](x, output_shape=cur_output_shape)
            # self.logger.debug(f"{x.shape} (block[{layer_idx}])")

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        if self.projection:
            x = self.projection(x)
            # self.logger.debug(f"{x.shape} (projection)")

        return x