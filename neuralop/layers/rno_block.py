import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..layers.spectral_convolution import SpectralConv
from ..layers.fno_block import FNOBlocks
from ..layers.complex import cselu


class RNOCell(nn.Module):
    """N-Dimensional Recurrent Neural Operator cell. The RNO cell takes in an
    input and history function, and it outputs the next step of the hidden function.

    The RNO cell implements the GRU-like recurrence relation with Fourier layers:
        z_t = σ(f1(x_t) + f2(h_{t-1}) + b1)            [update gate]
        r_t = σ(f3(x_t) + f4(h_{t-1}) + b2)            [reset gate]
        h̃_t = selu(f5(x_t) + f6(r_t ⊙ h_{t-1}) + b3)  [candidate state]
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t         [next state]

    where σ is the sigmoid function, ⊙ is element-wise multiplication,
    and f1-f6 are FNO blocks (Fourier neural operators).

    Parameters
    ----------
    n_modes : int tuple
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the RNO is inferred from ``len(n_modes)``
    hidden_channels : int
        number of hidden channels in the RNO

    Other Parameters
    ----------------
    resolution_scaling_factor : Union[Number, List[Number]], optional
        Factor by which to scale outputs for super-resolution, by default None
    max_n_modes : int or List[int], optional
        Maximum number of modes to keep along each dimension, by default None
    fno_block_precision : str, optional
        Floating point precision to use for computations. Options: "full", "half", "mixed", by default "full"
    use_channel_mlp : bool, optional
        Whether to use an MLP layer after each FNO block, by default True
    channel_mlp_dropout : float, optional
        Dropout parameter for self.channel_mlp, by default 0
    channel_mlp_expansion : float, optional
        Expansion parameter for self.channel_mlp, by default 0.5
    non_linearity : torch.nn.F module, optional
        Nonlinear activation function to use between layers, by default F.gelu
    stabilizer : Literal["tanh"], optional
        Stabilizing module to use between certain layers. Options: "tanh", None, by default None
    norm : Literal["ada_in", "group_norm", "instance_norm", "batch_norm"], optional
        Normalization layer to use. Options: "ada_in", "group_norm", "instance_norm", "batch_norm", None, by default None
    ada_in_features : int, optional
        Number of features for adaptive instance norm above, by default None
    preactivation : bool, optional
        Whether to call forward pass with pre-activation, by default False
        If True, call nonlinear activation and norm before Fourier convolution
        If False, call activation and norms after Fourier convolutions
    fno_skip : str, optional
        Module to use for FNO skip connections. Options: "linear", "soft-gating", "identity", None, by default "linear"
        If None, no skip connection is added. See layers.skip_connections for more details
    channel_mlp_skip : str, optional
        Module to use for ChannelMLP skip connections. Options: "linear", "soft-gating", "identity", None, by default "soft-gating"
        If None, no skip connection is added. See layers.skip_connections for more details
    complex_data : bool, optional
        Whether the FNO's data takes on complex values in space, by default False
    separable : bool, optional
        Separable parameter for SpectralConv, by default False
    factorization : str, optional
        Factorization parameter for SpectralConv. Options: "tucker", "cp", "tt", None, by default None
    rank : float, optional
        Rank parameter for SpectralConv, by default 1.0
    conv_module : BaseConv, optional
        Module to use for convolutions in FNO block, by default SpectralConv
    joint_factorization : bool, optional
        Whether to factorize all spectralConv weights as one tensor, by default False
    fixed_rank_modes : bool, optional
        Fixed_rank_modes parameter for SpectralConv, by default False
    implementation : str, optional
        Implementation parameter for SpectralConv. Options: "factorized", "reconstructed", by default "factorized"
    decomposition_kwargs : dict, optional
        Kwargs for tensor decomposition in SpectralConv, by default dict()

    References
    ----------
    .. [1] Paper: https://arxiv.org/abs/2308.08794
    """

    def __init__(
        self,
        n_modes,
        hidden_channels,
        resolution_scaling_factor=None,
        max_n_modes=None,
        fno_block_precision="full",
        use_channel_mlp=True,
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        ada_in_features=None,
        preactivation=False,
        fno_skip="linear",
        channel_mlp_skip="soft-gating",
        complex_data=False,
        separable=False,
        factorization=None,
        rank=1.0,
        conv_module=SpectralConv,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        scaling_factor = (
            None if not resolution_scaling_factor else [resolution_scaling_factor]
        )
        fno_kwargs = {
            "n_layers": 1,
            "max_n_modes": max_n_modes,
            "fno_block_precision": fno_block_precision,
            "use_channel_mlp": use_channel_mlp,
            "channel_mlp_dropout": channel_mlp_dropout,
            "channel_mlp_expansion": channel_mlp_expansion,
            "non_linearity": non_linearity,
            "stabilizer": stabilizer,
            "norm": norm,
            "preactivation": preactivation,
            "fno_skip": fno_skip,
            "channel_mlp_skip": channel_mlp_skip,
            "complex_data": complex_data,
            "separable": separable,
            "factorization": factorization,
            "rank": rank,
            "conv_module": conv_module,
            "fixed_rank_modes": fixed_rank_modes,
            "implementation": implementation,
            "decomposition_kwargs": decomposition_kwargs,
        }

        # For super-resolution: the hidden state h is always stored at the scaled resolution,
        # while input x remains at the original resolution. Therefore, only f1, f3, and f5
        # (which process x) need resolution scaling, while f2, f4, and f6 (which process h)
        # operate at the already-scaled resolution.

        self.input_gates = nn.ModuleList()
        self.hidden_gates = nn.ModuleList()
        self.biases = nn.ParameterList()

        # 3 gates: Update (z), Reset (r), Candidate (h_tilde)
        # Each gate involves one FNOBlock processing x (input_gate)
        # and one FNOBlock processing h (hidden_gate)
        for _ in range(3):
            self.input_gates.append(
                FNOBlocks(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    n_modes=n_modes,
                    resolution_scaling_factor=scaling_factor,
                    **fno_kwargs,
                )
            )
            self.hidden_gates.append(
                FNOBlocks(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    n_modes=n_modes,
                    resolution_scaling_factor=None,
                    **fno_kwargs,
                )
            )
            if complex_data:
                self.biases.append(nn.Parameter(torch.randn(()) + 1j * torch.randn(())))
            else:
                self.biases.append(nn.Parameter(torch.randn(())))

    def forward(self, x, h):
        """Forward pass for RNO cell.

        Parameters
        ----------
        x : torch.Tensor
            Input function at current timestep with shape (batch, hidden_channels, *spatial_dims)
        h : torch.Tensor
            Hidden state from previous timestep with shape (batch, hidden_channels, *spatial_dims_h)
            where spatial_dims_h may differ from spatial_dims if resolution_scaling_factor is set

        Returns
        -------
        torch.Tensor
            Updated hidden state with shape (batch, hidden_channels, *spatial_dims_h)
        """
        # Update gate
        update_gate = torch.sigmoid(
            self.input_gates[0](x) + self.hidden_gates[0](h) + self.biases[0]
        )

        # Reset gate
        reset_gate = torch.sigmoid(
            self.input_gates[1](x) + self.hidden_gates[1](h) + self.biases[1]
        )

        # Candidate state
        h_combined = self.input_gates[2](x) + self.hidden_gates[2](reset_gate * h) + self.biases[2]
        
        if x.dtype == torch.cfloat:
            candidate_state = cselu(h_combined)  # complex SELU for complex data
        else:
            candidate_state = F.selu(h_combined)  # regular SELU for real data

        h_next = (1.0 - update_gate) * h + update_gate * candidate_state

        return h_next


class RNOBlock(nn.Module):
    """N-Dimensional Recurrent Neural Operator layer. The RNO layer extends the
    action of the RNO cell to take in some sequence of time-steps as input
    and output the next output function.

    The layer applies the RNO cell recurrently over a sequence of inputs:
        For t = 1 to T:
            h_t = RNOCell(x_t, h_{t-1})

    where the cell implements:
        z_t = σ(f1(x_t) + f2(h_{t-1}) + b1)            [update gate]
        r_t = σ(f3(x_t) + f4(h_{t-1}) + b2)            [reset gate]
        h̃_t = selu(f5(x_t) + f6(r_t ⊙ h_{t-1}) + b3)  [candidate state]
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t         [next state]

    Parameters
    ----------
    n_modes : int tuple
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the RNO is inferred from ``len(n_modes)``
    hidden_channels : int
        number of hidden channels in the RNO
    return_sequences : boolean, optional
        Whether to return the sequence of hidden states associated with processing
        the inputs sequence of functions. Default: False

    Other Parameters
    ----------------
    resolution_scaling_factor : Union[Number, List[Number]], optional
        Factor by which to scale outputs for super-resolution, by default None
    max_n_modes : int or List[int], optional
        Maximum number of modes to keep along each dimension, by default None
    fno_block_precision : str, optional
        Floating point precision to use for computations. Options: "full", "half", "mixed", by default "full"
    use_channel_mlp : bool, optional
        Whether to use an MLP layer after each FNO block, by default True
    channel_mlp_dropout : float, optional
        Dropout parameter for self.channel_mlp, by default 0
    channel_mlp_expansion : float, optional
        Expansion parameter for self.channel_mlp, by default 0.5
    non_linearity : torch.nn.F module, optional
        Nonlinear activation function to use between layers, by default F.gelu
    stabilizer : Literal["tanh"], optional
        Stabilizing module to use between certain layers. Options: "tanh", None, by default None
    norm : Literal["ada_in", "group_norm", "instance_norm", "batch_norm"], optional
        Normalization layer to use. Options: "ada_in", "group_norm", "instance_norm", "batch_norm", None, by default None
    ada_in_features : int, optional
        Number of features for adaptive instance norm above, by default None
    preactivation : bool, optional
        Whether to call forward pass with pre-activation, by default False
        If True, call nonlinear activation and norm before Fourier convolution
        If False, call activation and norms after Fourier convolutions
    fno_skip : str, optional
        Module to use for FNO skip connections. Options: "linear", "soft-gating", "identity", None, by default "linear"
        If None, no skip connection is added. See layers.skip_connections for more details
    channel_mlp_skip : str, optional
        Module to use for ChannelMLP skip connections. Options: "linear", "soft-gating", "identity", None, by default "soft-gating"
        If None, no skip connection is added. See layers.skip_connections for more details
    complex_data : bool, optional
        Whether the FNO's data takes on complex values in space, by default False
    separable : bool, optional
        Separable parameter for SpectralConv, by default False
    factorization : str, optional
        Factorization parameter for SpectralConv. Options: "tucker", "cp", "tt", None, by default None
    rank : float, optional
        Rank parameter for SpectralConv, by default 1.0
    conv_module : BaseConv, optional
        Module to use for convolutions in FNO block, by default SpectralConv
    joint_factorization : bool, optional
        Whether to factorize all spectralConv weights as one tensor, by default False
    fixed_rank_modes : bool, optional
        Fixed_rank_modes parameter for SpectralConv, by default False
    implementation : str, optional
        Implementation parameter for SpectralConv. Options: "factorized", "reconstructed", by default "factorized"
    decomposition_kwargs : dict, optional
        Kwargs for tensor decomposition in SpectralConv, by default dict()

    References
    ----------
    .. [1] Paper: https://arxiv.org/abs/2308.08794
    """

    def __init__(
        self,
        n_modes,
        hidden_channels,
        return_sequences=False,
        resolution_scaling_factor=None,
        max_n_modes=None,
        fno_block_precision="full",
        use_channel_mlp=True,
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        ada_in_features=None,
        preactivation=False,
        fno_skip="linear",
        channel_mlp_skip="soft-gating",
        complex_data=False,
        separable=False,
        factorization=None,
        rank=1.0,
        conv_module=SpectralConv,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.return_sequences = return_sequences
        self.resolution_scaling_factor = resolution_scaling_factor

        self.cell = RNOCell(
            n_modes,
            hidden_channels,
            resolution_scaling_factor=resolution_scaling_factor,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
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
            separable=separable,
            factorization=factorization,
            rank=rank,
            conv_module=conv_module,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
        )
        if complex_data:
            self.bias_h = nn.Parameter(torch.randn(()) + 1j * torch.randn(()))
        else:
            self.bias_h = nn.Parameter(torch.randn(()))

    def forward(self, x, h=None):
        """Forward pass for RNO layer.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence with shape (batch, timesteps, hidden_channels, *spatial_dims)
        h : torch.Tensor, optional
            Initial hidden state with shape (batch, hidden_channels, *spatial_dims_h).
            If None, initialized to zeros with added bias. Default: None

        Returns
        -------
        torch.Tensor
            If return_sequences=True: hidden states for all timesteps with shape
                (batch, timesteps, hidden_channels, *spatial_dims_h)
            If return_sequences=False: final hidden state with shape
                (batch, hidden_channels, *spatial_dims_h)
        """
        batch_size, timesteps, dim = x.shape[:3]
        dom_sizes = x.shape[3:]

        # Initialize hidden state if not provided
        if h is None:
            # Compute the spatial dimensions for h (scaled if resolution_scaling_factor is set)
            if not self.resolution_scaling_factor:
                h_shape = (batch_size, self.hidden_channels, *dom_sizes)
            else:
                scaled_sizes = tuple(
                    [int(round(self.resolution_scaling_factor * s)) for s in dom_sizes]
                )
                h_shape = (batch_size, self.hidden_channels, *scaled_sizes)
            h = torch.zeros(h_shape, dtype=x.dtype).to(x.device)
            h += self.bias_h

        outputs = []
        # Process each timestep sequentially through the RNO cell
        for i in range(timesteps):
            h = self.cell(x[:, i], h)  # Update hidden state with current input
            if self.return_sequences:
                outputs.append(h)

        if self.return_sequences:
            return torch.stack(outputs, dim=1)
        else:
            return h
