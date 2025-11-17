import torch.nn as nn
import torch.nn.functional as F
import torch

# Set warning filter to show each warning only once
import warnings

warnings.filterwarnings("once", category=UserWarning)

from ..layers.channel_mlp import ChannelMLP
from ..layers.spectral_convolution import SpectralConv
from ..layers.skip_connections import skip_connection
from ..layers.padding import DomainPadding
from ..layers.fno_block import FNOBlocks
from ..layers.resample import resample
from ..layers.embeddings import GridEmbedding2D, GridEmbeddingND


class UNO(nn.Module):
    """U-Shaped Neural Operator
    
    The architecture is described in  [1]_.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Determined by the problem.
    out_channels : int
        Number of output channels. Determined by the problem.
    hidden_channels : int
        Initial width of the UNO. This significantly affects the number of parameters of the UNO.
        Good starting point can be 64, and then increased if more expressivity is needed.
        Update lifting_channels and projection_channels accordingly since they are proportional to hidden_channels.
    uno_out_channels : list
        Number of output channels of each Fourier layer.
        Example: For a five layer UNO uno_out_channels can be [32,64,64,64,32]
    uno_n_modes : list
        Number of Fourier modes to use in integral operation of each Fourier layer (along each dimension).
        Example: For a five layer UNO with 2D input the uno_n_modes can be: [[5,5],[5,5],[5,5],[5,5],[5,5]]
    uno_scalings : list
        Scaling factors for each Fourier layer.
        Example: For a five layer UNO with 2D input, the uno_scalings can be: [[1.0,1.0],[0.5,0.5],[1,1],[1,1],[2,2]]
    n_layers : int, optional
        Number of Fourier layers. Default: 4
    lifting_channels : int, optional
        Number of hidden channels of the lifting block of the FNO. Default: 256
    projection_channels : int, optional
        Number of hidden channels of the projection block of the FNO. Default: 256
    positional_embedding : Union[str, GridEmbedding2D, GridEmbeddingND, None], optional
        Positional embedding to apply to last channels of raw input before being passed through the UNO.
        Options:
        - "grid": Appends a grid positional embedding with default settings to the last channels of raw input.
          Assumes the inputs are discretized over a grid with entry [0,0,...] at the origin and side lengths of 1.
        - GridEmbedding2D: Uses this module directly for 2D cases.
        - GridEmbeddingND: Uses this module directly (see `neuralop.embeddings.GridEmbeddingND` for details).
        - None: Does nothing.
        Default: "grid"
    horizontal_skips_map : Dict, optional
        A dictionary {b: a, ...} denoting horizontal skip connection from a-th layer to b-th layer.
        If None, default skip connection is applied.
        Example: For a 5 layer UNO architecture, the skip connections can be horizontal_skips_map = {4:0,3:1}
        Default: None
    channel_mlp_dropout : float, optional
        Dropout parameter for ChannelMLP after each FNO block. Default: 0
    channel_mlp_expansion : float, optional
        Expansion parameter for ChannelMLP after each FNO block. Default: 0.5
    non_linearity : nn.Module, optional
        Non-linearity module to use. Default: F.gelu
    norm : str, optional
        Normalization layer to use. Options: "ada_in", "group_norm", "instance_norm", None. Default: None
    preactivation : bool, optional
        Whether to use ResNet-style preactivation. Default: False
    fno_skip : str, optional
        Type of skip connection to use in FNO layers. Options: "linear", "identity", "soft-gating", None.
        Default: "linear"
    horizontal_skip : str, optional
        Type of skip connection to use in horizontal connections. Options: "linear", "identity", "soft-gating", None.
        Default: "linear"
    channel_mlp_skip : str, optional
        Type of skip connection to use in channel-mixing MLP. Options: "linear", "identity", "soft-gating", None.
        Default: "soft-gating"
    separable : bool, optional
        Whether to use a separable spectral convolution. Default: False
    factorization : str, optional
        Tensor factorization of the parameters weight to use.
        Options: "None", "Tucker", "CP", "TT"
        Other factorization methods supported by tltorch. Default: None
    rank : float, optional
        Rank of the tensor factorization of the Fourier weights. Default: 1.0.
        Set to float <1.0 when using TFNO (i.e. when factorization is not None).
        A TFNO with rank 0.1 has roughly 10% of the parameters of a dense FNO.
    fixed_rank_modes : bool, optional
        Whether to not factorize certain modes. Default: False
    implementation : str, optional
        If factorization is not None, forward mode to use.
        Options: "reconstructed", "factorized". Default: "factorized"
    decomposition_kwargs : dict, optional
        Additional parameters to pass to the tensor decomposition. Default: {}
    domain_padding : Union[float, List[float], None], optional
        Percentage of padding to use. If not None, percentage of padding to use. Default: None
    fft_norm : str, optional
        FFT normalization mode. Default: "forward"

    References
    -----------
    .. [1] :

    Rahman, M.A., Ross, Z., Azizzadenesheli, K. "U-NO: U-shaped
        Neural Operators" (2022). TMLR 2022, https://arxiv.org/pdf/2204.11127.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        lifting_channels=256,
        projection_channels=256,
        positional_embedding="grid",
        n_layers=4,
        uno_out_channels=None,
        uno_n_modes=None,
        uno_scalings=None,
        horizontal_skips_map=None,
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        norm=None,
        preactivation=False,
        fno_skip="linear",
        horizontal_skip="linear",
        channel_mlp_skip="soft-gating",
        separable=False,
        factorization=None,
        rank=1.0,
        fixed_rank_modes=False,
        integral_operator=SpectralConv,
        operator_block=FNOBlocks,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        verbose=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        assert uno_out_channels is not None, "uno_out_channels can not be None"
        assert uno_n_modes is not None, "uno_n_modes can not be None"
        assert uno_scalings is not None, "uno_scalings can not be None"
        assert (
            len(uno_out_channels) == n_layers
        ), "Output channels for all layers are not given"
        assert (
            len(uno_n_modes) == n_layers
        ), "number of modes for all layers are not given"
        assert (
            len(uno_scalings) == n_layers
        ), "Scaling factor for all layers are not given"

        self.n_dim = len(uno_n_modes[0])
        self.uno_out_channels = uno_out_channels
        self.uno_n_modes = uno_n_modes
        self.uno_scalings = uno_scalings
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.horizontal_skips_map = horizontal_skips_map
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = (fno_skip,)
        self.channel_mlp_skip = (channel_mlp_skip,)
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation

        self.operator_block = operator_block
        self.integral_operator = integral_operator

        # create positional embedding at the beginning of the model
        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0.0, 1.0]] * self.n_dim
            self.positional_embedding = GridEmbeddingND(
                in_channels=self.in_channels,
                dim=self.n_dim,
                grid_boundaries=spatial_grid_boundaries,
            )
        elif isinstance(positional_embedding, GridEmbedding2D):
            if self.n_dim == 2:
                self.positional_embedding = positional_embedding
            else:
                raise ValueError(f'Error: expected {self.n_dim}-d positional embeddings, got {positional_embedding}')
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
        elif positional_embedding == None:
            self.positional_embedding = None
        else:
            raise ValueError(
                f"Error: tried to instantiate FNO positional embedding with {positional_embedding},\
                              expected one of 'grid', GridEmbeddingND"
            )

        if self.positional_embedding is not None:
            in_channels += self.n_dim

        # constructing default skip maps
        if self.horizontal_skips_map is None:
            self.horizontal_skips_map = {}
            for i in range(
                0,
                n_layers // 2,
            ):
                # example, if n_layers = 5, then 4:0, 3:1
                self.horizontal_skips_map[n_layers - i - 1] = i
        # self.uno_scalings may be a 1d list specifying uniform scaling factor at each layer
        # or a 2d list, where each row specifies scaling factors along each dimention.
        # To get the final (end to end) scaling factors we need to multiply
        # the scaling factors (a list) of all layer.

        self.end_to_end_scaling_factor = [1] * len(self.uno_scalings[0])
        # multiplying scaling factors
        for k in self.uno_scalings:
            self.end_to_end_scaling_factor = [
                i * j for (i, j) in zip(self.end_to_end_scaling_factor, k)
            ]

        # list with a single element is replaced by the scaler.
        if len(self.end_to_end_scaling_factor) == 1:
            self.end_to_end_scaling_factor = self.end_to_end_scaling_factor[0]

        if isinstance(self.end_to_end_scaling_factor, (float, int)):
            self.end_to_end_scaling_factor = [
                self.end_to_end_scaling_factor
            ] * self.n_dim

        if verbose:
            print("calculated out factor", self.end_to_end_scaling_factor)

        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                resolution_scaling_factor=self.end_to_end_scaling_factor,
            )
        else:
            self.domain_padding = None

        self.lifting = ChannelMLP(
            in_channels=in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=self.lifting_channels,
            n_layers=2,
            n_dim=self.n_dim,
        )
        self.fno_blocks = nn.ModuleList([])
        self.horizontal_skips = torch.nn.ModuleDict({})
        prev_out = self.hidden_channels

        for i in range(self.n_layers):
            if i in self.horizontal_skips_map.keys():
                prev_out = (
                    prev_out + self.uno_out_channels[self.horizontal_skips_map[i]]
                )

            self.fno_blocks.append(
                self.operator_block(
                    in_channels=prev_out,
                    out_channels=self.uno_out_channels[i],
                    n_modes=self.uno_n_modes[i],
                    channel_mlp_dropout=channel_mlp_dropout,
                    channel_mlp_expansion=channel_mlp_expansion,
                    resolution_scaling_factor=[self.uno_scalings[i]],
                    non_linearity=non_linearity,
                    norm=norm,
                    preactivation=preactivation,
                    fno_skip=fno_skip,
                    channel_mlp_skip=channel_mlp_skip,
                    rank=rank,
                    fixed_rank_modes=fixed_rank_modes,
                    implementation=implementation,
                    separable=separable,
                    factorization=factorization,
                    decomposition_kwargs=decomposition_kwargs,
                )
            )

            if i in self.horizontal_skips_map.values():
                self.horizontal_skips[str(i)] = skip_connection(
                    self.uno_out_channels[i],
                    self.uno_out_channels[i],
                    skip_type=horizontal_skip,
                    n_dim=self.n_dim,
                )

            prev_out = self.uno_out_channels[i]

        self.projection = ChannelMLP(
            in_channels=prev_out,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

    def forward(self, x, **kwargs):
        if kwargs:
            warnings.warn(
                f"UNO.forward() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)
        output_shape = [
            int(round(i * j))
            for (i, j) in zip(x.shape[-self.n_dim :], self.end_to_end_scaling_factor)
        ]

        skip_outputs = {}
        cur_output = None
        for layer_idx in range(self.n_layers):
            if layer_idx in self.horizontal_skips_map.keys():
                skip_val = skip_outputs[self.horizontal_skips_map[layer_idx]]
                resolution_scaling_factors = [
                    m / n for (m, n) in zip(x.shape, skip_val.shape)
                ]
                resolution_scaling_factors = resolution_scaling_factors[-1 * self.n_dim :]
                t = resample(
                    skip_val, resolution_scaling_factors, list(range(-self.n_dim, 0))
                )
                x = torch.cat([x, t], dim=1)

            if layer_idx == self.n_layers - 1:
                cur_output = output_shape
            x = self.fno_blocks[layer_idx](x, output_shape=cur_output)

            if layer_idx in self.horizontal_skips_map.values():
                skip_outputs[layer_idx] = self.horizontal_skips[str(layer_idx)](x)

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)
        return x
