from typing import Tuple, List, Union, Literal

Number = Union[float, int]

import torch
import torch.nn as nn
import torch.nn.functional as F

# Set warning filter to show each warning only once
import warnings

warnings.filterwarnings("once", category=UserWarning)

from ..layers.rno_block import RNOBlock
from ..layers.padding import DomainPadding
from ..layers.fno_block import FNOBlocks
from ..layers.channel_mlp import ChannelMLP
from ..layers.spectral_convolution import SpectralConv
from ..layers.complex import ComplexValued
from ..layers.embeddings import GridEmbeddingND, GridEmbedding2D
from .base_model import BaseModel


class RNO(BaseModel, name="RNO"):
    """
    N-Dimensional Recurrent Neural Operator.

    The RNO has an identical architecture to the finite-dimensional GRU, with
    the exception that linear matrix-vector multiplications are replaced by
    Fourier layers (Li et al., 2021), and for regression problems, the output
    nonlinearity is replaced by a SELU activation.

    The operation of the GRU is as follows:
    z_t = sigmoid(W_z x + U_z h_{t-1} + b_z)
    r_t = sigmoid(W_r x + U_r h_{t-1} + b_r)
    \hat h_t = selu(W_h x_t + U_h (r_t * h_{t-1}) + b_h)
    h_t = (1 - z_t) * h_{t-1} + z_t * \hat h_t,

    where * is element-wise, the b_i's are bias functions, and W_i, U_i are
    linear Fourier layers.

    Paper:
    .. [RNO] Liu-Schiaffini, M., Singer, C. E., Kovachki, N., Schneider, T.,
        Azizzadenesheli, K., & Anandkumar, A. (2023). Tipping point forecasting
        in non-stationary dynamics on function spaces. arXiv preprint
        arXiv:2308.08794.

    Main Parameters
    ---------------
    n_modes : Tuple[int, ...]
        Number of modes to keep in Fourier Layer, along each dimension.
        The dimensionality of the RNO is inferred from ``len(n_modes)``.
        n_modes must be larger enough but smaller than max_resolution//2 (Nyquist frequency)
    in_channels : int
        Number of input channels in input function. Determined by the problem.
    out_channels : int
        Number of output channels in output function. Determined by the problem.
    hidden_channels : int
        Width of the RNO (i.e. number of channels).
        This significantly affects the number of parameters of the RNO.
        For 1D problems, 24-48 channels are typically sufficient. For 2D/3D problems
        without Tucker/CP/TT factorization, start with 16-32 channels to avoid
        excessive parameters. Increase if more expressivity is needed.
        Update lifting_channel_ratio and projection_channel_ratio accordingly since they are proportional to hidden_channels.
    n_layers : int, optional
        Number of RNO layers to use. Default: 4
    rno_skip : bool, optional
        Whether to use skip connections between RNO layers. When True, adds
        the input to each layer's output: x_{l+1} = x_l + RNOBlock(x_l).
        Default: False. Unlike FNO where skip connections improve gradient flow,
        RNO's recurrent structure already provides temporal gradient pathways,
        so skip connections are optional and may not always improve performance.

    Other parameters
    ---------------
    lifting_channel_ratio : Number, optional
        Ratio of lifting channels to hidden_channels.
        The number of lifting channels in the lifting block of the RNO is
        lifting_channel_ratio * hidden_channels (e.g. default 2 * hidden_channels).
    projection_channel_ratio : Number, optional
        Ratio of projection channels to hidden_channels.
        The number of projection channels in the projection block of the RNO is
        projection_channel_ratio * hidden_channels (e.g. default 2 * hidden_channels).
    positional_embedding : Union[str, nn.Module], optional
        Type of positional embedding to use. Options:
        - "grid": Use default grid-based positional embedding
        - GridEmbeddingND: Custom N-dimensional grid embedding
        - GridEmbedding2D: Custom 2D grid embedding (only for 2D problems)
        - None: No positional embedding
        Default: None
    non_linearity : nn.Module, optional
        Non-Linear activation function module to use. Default: F.gelu
    norm : Literal["ada_in", "group_norm", "instance_norm"], optional
        Normalization layer to use. Options: "ada_in", "group_norm", "instance_norm", None. Default: None
    complex_data : bool, optional
        Whether the data is complex-valued. If True, initializes complex-valued modules. Default: False
    use_channel_mlp : bool, optional
        Whether to use an MLP layer after each FNO block. Default: True
    channel_mlp_dropout : float, optional
        Dropout parameter for ChannelMLP in FNO Block. Default: 0
    channel_mlp_expansion : float, optional
        Expansion parameter for ChannelMLP in FNO Block. Default: 0.5
    channel_mlp_skip : Literal["linear", "identity", "soft-gating", None], optional
        Type of skip connection to use in channel-mixing mlp. Options: "linear", "identity", "soft-gating", None.
        Default: "soft-gating"
    fno_skip : Literal["linear", "identity", "soft-gating", None], optional
        Type of skip connection to use in FNO layers. Options: "linear", "identity", "soft-gating", None.
        Default: "linear"
    return_sequences : bool, optional
        Whether the final RNO layer returns the full sequence of hidden states or just the final state.
        Intermediate layers always return sequences. Default: False
    resolution_scaling_factor : Union[Number, List[Number]], optional
        Layer-wise factor by which to scale the domain resolution of function.
        Options:
        - None: No scaling
        - Single number n: Scales resolution by n at each layer
        - List of numbers [n_0, n_1,...]: Scales layer i's resolution by n_i
        Default: None
    domain_padding : Union[Number, List[Number]], optional
        Percentage of padding to use.
        Options:
        - None: No padding
        - Single number: Percentage of padding to use along all dimensions
        - List of numbers [p1, p2, ..., pN]: Percentage of padding along each dimension
        Default: None
    fno_block_precision : str, optional
        Precision mode in which to perform spectral convolution.
        Options: "full", "half", "mixed". Default: "full".
    stabilizer : str, optional
        Whether to use a stabilizer in FNO block. Options: "tanh", None. Default: None.
        stabilizer greatly improves performance in the case `fno_block_precision='mixed'`.
    max_n_modes : Tuple[int, ...], optional
        Maximum number of modes to use in Fourier domain during training.
        None means that all the n_modes are used.
        Tuple of integers: Incrementally increase the number of modes during training.
        This can be updated dynamically during training.
    factorization : str, optional
        Tensor factorization of the FNO layer weights to use.
        Options: "None", "Tucker", "CP", "TT"
        Other factorization methods supported by tltorch. Default: None
    rank : float, optional
        Tensor rank to use in factorization. Default: 1.0
        Set to float <1.0 when using TFNO (i.e. when factorization is not None).
        A TFNO with rank 0.1 has roughly 10% of the parameters of a dense FNO.
    fixed_rank_modes : bool, optional
        Whether to not factorize certain modes. Default: False
    implementation : str, optional
        Implementation method for factorized tensors.
        Options: "factorized", "reconstructed". Default: "factorized"
    decomposition_kwargs : dict, optional
        Extra kwargs for tensor decomposition (see `tltorch.FactorizedTensor`). Default: {}
    separable : bool, optional
        Whether to use a separable spectral convolution. Default: False
    preactivation : bool, optional
        Whether to compute FNO forward pass with resnet-style preactivation. Default: False
    conv_module : nn.Module, optional
        Module to use for FNOBlock's convolutions. Default: SpectralConv
    """

    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        rno_skip: bool = False,
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Literal["ada_in", "group_norm", "instance_norm"] = None,
        complex_data: bool = False,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        return_sequences: bool = False,
        resolution_scaling_factor: Union[Number, List[Number]] = None,
        domain_padding: Union[Number, List[Number]] = None,
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int, ...] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = None,
        separable: bool = False,
        preactivation: bool = False,
        conv_module: nn.Module = SpectralConv,
    ):
        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        super().__init__()
        self.n_dim = len(n_modes)

        # n_modes is a special property - see the class' property for underlying mechanism
        # When updated, change should be reflected in fno blocks
        self.n_modes = n_modes

        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.rno_skip = rno_skip

        # init lifting and projection channels using ratios w.r.t hidden channels
        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = int(lifting_channel_ratio * self.hidden_channels)

        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = int(projection_channel_ratio * self.hidden_channels)

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
        self.complex_data = complex_data
        self.fno_block_precision = fno_block_precision

        ## Domain padding
        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            self.domain_padding = None

        ## Positional embedding
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
                raise ValueError(
                    f"Error: expected {self.n_dim}-d positional embeddings, got {positional_embedding}"
                )
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
        elif positional_embedding is None:
            self.positional_embedding = None
        else:
            raise ValueError(
                f"Error: tried to instantiate RNO positional embedding with {positional_embedding}, "
                f"expected one of 'grid', GridEmbeddingND, GridEmbedding2D, or None"
            )

        ## Resolution scaling factor
        if resolution_scaling_factor:
            if isinstance(resolution_scaling_factor, (float, int)):
                resolution_scaling_factor = [resolution_scaling_factor] * self.n_layers
        else:
            resolution_scaling_factor = [None] * self.n_layers
        self.resolution_scaling_factor = resolution_scaling_factor

        ## Return sequences configuration
        # Intermediate layers always return sequences, only last layer is configurable
        return_sequences_list = [True] * (self.n_layers - 1) + [return_sequences]
        self.return_sequences = return_sequences_list

        module_list = [
            RNOBlock(
                n_modes=self.n_modes,
                hidden_channels=hidden_channels,
                return_sequences=return_sequences_list[i],
                resolution_scaling_factor=self.resolution_scaling_factor[i],
                use_channel_mlp=use_channel_mlp,
                channel_mlp_dropout=channel_mlp_dropout,
                channel_mlp_expansion=channel_mlp_expansion,
                non_linearity=non_linearity,
                stabilizer=stabilizer,
                norm=norm,
                preactivation=preactivation,
                fno_skip=fno_skip,
                channel_mlp_skip=channel_mlp_skip,
                complex_data=complex_data,
                max_n_modes=max_n_modes,
                fno_block_precision=fno_block_precision,
                rank=rank,
                fixed_rank_modes=fixed_rank_modes,
                implementation=implementation,
                separable=separable,
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                conv_module=conv_module,
            )
            for i in range(n_layers)
        ]
        self.layers = nn.ModuleList(module_list)

        ## Lifting layer
        # if adding a positional embedding, add those channels to lifting
        lifting_in_channels = self.in_channels
        if self.positional_embedding is not None:
            lifting_in_channels += self.n_dim
        # if lifting_channels is set, make lifting a Channel-Mixing MLP
        # with a hidden layer of size lifting_channels
        if self.lifting_channels:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
                non_linearity=non_linearity,
            )
        # otherwise, make it a linear layer
        else:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
                non_linearity=non_linearity,
            )
        # Convert lifting to a complex ChannelMLP if self.complex_data==True
        if self.complex_data:
            self.lifting = ComplexValued(self.lifting)

        ## Projection layer
        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        if self.complex_data:
            self.projection = ComplexValued(self.projection)

    def forward(
        self,
        x,
        init_hidden_states=None,
        return_hidden_states=False,
        keep_states_padded=False,
    ):
        """
        Forward pass for the Recurrent Neural Operator.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, timesteps, in_channels, *spatial_dims),
            where len(spatial_dims) == self.n_dim. The channel dimension MUST be at
            index 2 and the time dimension MUST be at index 1.
        init_hidden_states : list[torch.Tensor] | None
            Optional list of per-layer initial hidden states. Each tensor should have
            shape (batch, hidden_channels, *spatial_dims_h), where spatial_dims_h are
            the unpadded spatial dimensions at each layer (accounting for resolution
            scaling). If domain_padding is enabled, hidden states will be automatically
            padded like the input x, unless keep_states_padded is True.
            If None, all hidden states are initialized internally.
        return_hidden_states : bool, optional
            Whether to return the final hidden states for each layer in addition to
            the prediction. Default: False
        keep_states_padded : bool, optional
            If True, treats provided init_hidden_states as already padded (no additional
            padding applied) and returns final_hidden_states in padded form (no unpadding).
            Use this for efficient autoregressive loops to preserve boundary information.
            Default: False

        Returns
        -------
        pred : torch.Tensor
            Output tensor with shape (batch, out_channels, *spatial_dims_out).
        final_hidden_states : list[torch.Tensor], optional
            Returned only if return_hidden_states=True.
        """
        # Strict input validation to avoid silent errors from resolution invariance
        expected_rank = 3 + self.n_dim
        if x.ndim != expected_rank:
            raise ValueError(
                f"RNO.forward expected input of rank {expected_rank} = "
                f"(batch, timesteps, channels, {self.n_dim} spatial dims), got rank {x.ndim} with shape {tuple(x.shape)}"
            )
        if x.shape[2] != self.in_channels:
            raise ValueError(
                f"RNO.forward expected x.shape[2] == in_channels ({self.in_channels}); "
                f"got {x.shape[2]}. Input must be shaped as (batch, timesteps, in_channels, *spatial_dims)."
            )
        # x shape (batch, timesteps, in_channels, *spatial_dims)
        batch_size, timesteps = x.shape[:2]
        dom_sizes = x.shape[3 : 3 + self.n_dim]

        if init_hidden_states is None:
            init_hidden_states = [None] * self.n_layers
        else:
            # If domain padding is enabled AND states are not already padded (keep_states_padded=False), pad them.
            if self.domain_padding and not keep_states_padded:
                padded_hidden_states = []
                for h in init_hidden_states:
                    if h is not None:
                        h = self.domain_padding.pad(h)
                    padded_hidden_states.append(h)
                init_hidden_states = padded_hidden_states

        # Reshape for processing: (batch*timesteps, channels, *spatial_dims)
        x = x.reshape(batch_size * timesteps, *x.shape[2:])

        # append spatial pos embedding if set
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)
        x = x.reshape(batch_size, timesteps, *x.shape[1:])

        if self.domain_padding:
            # DomainPadding expects (batch, channels, *spatial_dims), so reshape to remove timestep dim
            x = x.reshape(batch_size * timesteps, *x.shape[2:])
            x = self.domain_padding.pad(x)
            x = x.reshape(batch_size, timesteps, *x.shape[1:])

        final_hidden_states = []
        for i in range(self.n_layers):
            pred_x = self.layers[i](x, init_hidden_states[i])
            if i < self.n_layers - 1:
                if self.rno_skip:
                    x = x + pred_x
                else:
                    x = pred_x
                final_hidden_states.append(x[:, -1])
            else:
                x = pred_x
                final_hidden_states.append(x)
        h = final_hidden_states[-1]

        if self.domain_padding:
            # DomainPadding.unpad expects (batch, channels, *spatial_dims)
            # h comes from the last layer, which is padded logic.

            # We ALWAYS unpad 'h' for the projection layer to produce the correct output size
            # (Computation continues in unpadded space for projection)
            h = self.domain_padding.unpad(h)

            # For final_hidden_states, we unpad UNLESS explicitly told to keep them padded
            if return_hidden_states and not keep_states_padded:
                unpadded_states = []
                for state in final_hidden_states:
                    unpadded_states.append(self.domain_padding.unpad(state))
                final_hidden_states = unpadded_states

        pred = self.projection(h)

        if return_hidden_states:
            return pred, final_hidden_states
        else:
            return pred

    def predict(self, x, num_steps, grid_function=None):
        """Autoregressively predict future time steps.

        Performs autoregressive rollout by iteratively feeding predictions back as input.
        At each step, the model predicts the next time step from the current input,
        then uses that prediction as input for the subsequent prediction.

        Parameters
        ----------
        x : torch.Tensor
            Initial input sequence with shape (batch, timesteps, in_channels, *spatial_dims).
            The last time step of this sequence serves as the starting point for predictions.
        num_steps : int
            Number of future time steps to predict autoregressively.
        grid_function : callable, optional
            Function that generates positional embeddings (e.g., spatial coordinates) to
            concatenate with predictions before feeding them back as input. Should have
            signature grid_function(shape, device) -> torch.Tensor, where the returned
            tensor has shape matching the input requirements. Use this when your model
            was trained with concatenated positional information. Default: None

        Returns
        -------
        torch.Tensor
            Predicted sequence with shape (batch, num_steps, out_channels, *spatial_dims),
            containing the autoregressively generated future states.

        Notes
        -----
        This method maintains hidden states across prediction steps, enabling the RNO
        to leverage its recurrent structure during autoregressive generation.
        """
        output = []
        states = [None] * self.n_layers

        for _ in range(num_steps):
            pred, states = self.forward(
                x, states, return_hidden_states=True, keep_states_padded=True
            )

            output.append(pred)
            x = pred.reshape((pred.shape[0], 1, *pred.shape[1:]))
            if grid_function:
                grid = grid_function(
                    (x.shape[0], x.shape[1], 1, x.shape[-2], x.shape[-1]), x.device
                )
                x = torch.cat((x, grid), dim=2)

        return torch.stack(output, dim=1)
