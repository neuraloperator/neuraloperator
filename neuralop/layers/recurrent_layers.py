import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..layers.spectral_convolution import SpectralConv
from ..layers.fno_block import FNOBlocks

class RNO_cell(nn.Module):
    """N-Dimensional Recurrent Neural Operator cell. The RNO cell takes in an
    input and history function, and it outputs the next step of the hidden function.
    
    Paper: https://arxiv.org/abs/2308.08794 

    Parameters
    ------------------
    n_modes : int tuple
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the RNO is inferred from ``len(n_modes)``
    width : int
        width of the RNO (i.e. number of channels)

    Other Parameters
    -------------------
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

    Other Parameters
    -------------------
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
    """
    def __init__(
        self, 
        n_modes, 
        width, 
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
        # resolution_scaling_factor is provided here as an integer or float
        super().__init__()
        self.width = width
        scaling_factor = None if not resolution_scaling_factor else [resolution_scaling_factor]
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

        # Some resolution_scaling_factors are None to super-resolution purposes. We use the hidden representation in the scaled size always (it's initialized that way),
        # so we only need to scale the dimensions of f1, f3, and f5, which act on x (original dimensionality).
        self.f1 = FNOBlocks(width, width, n_modes, resolution_scaling_factor=scaling_factor, 
                            **fno_kwargs)
        self.f2 = FNOBlocks(width, width, n_modes, resolution_scaling_factor=None, **fno_kwargs)
        self.f3 = FNOBlocks(width, width, n_modes, resolution_scaling_factor=scaling_factor, **fno_kwargs)
        self.f4 = FNOBlocks(width, width, n_modes, resolution_scaling_factor=None, **fno_kwargs)
        self.f5 = FNOBlocks(width, width, n_modes, resolution_scaling_factor=scaling_factor, **fno_kwargs)
        self.f6 = FNOBlocks(width, width, n_modes, resolution_scaling_factor=None, **fno_kwargs)

        self.b1 = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.))) # constant bias terms
        self.b2 = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.)))
        self.b3 = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.)))
    
    def forward(self, x, h):
        z = torch.sigmoid(self.f1(x) + self.f2(h) + self.b1)
        r = torch.sigmoid(self.f3(x) + self.f4(h) + self.b2)
        h_hat = F.selu(self.f5(x) + self.f6(r * h) + self.b3) # selu for regression problem

        h_next = (1. - z) * h + z * h_hat

        return h_next

class RNO_layer(nn.Module):
    """N-Dimensional Recurrent Neural Operator layer. The RNO layer extends the
    action of the RNO cell to take in some sequence of time-steps as input
    and output the next output function. 

    Paper: https://arxiv.org/abs/2308.08794 

    Parameters
    ------------------
    n_modes : int tuple
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the RNO is inferred from ``len(n_modes)``
    width : int
        width of the RNO (i.e. number of channels)
    return_sequences : boolean, optional
        Whether to return the sequence of hidden states associated with processing
        the inputs sequence of functions.

    Other Parameters
    -------------------
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

    Other Parameters
    -------------------
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
    """
    def __init__(
        self, 
        n_modes, 
        width, 
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

        self.width = width
        self.return_sequences = return_sequences
        self.resolution_scaling_factor = resolution_scaling_factor

        self.cell = RNO_cell(
            n_modes, 
            width, 
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
        self.bias_h = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.)))

    def forward(self, x, h=None):
        batch_size, timesteps, dim = x.shape[:3]
        dom_sizes = x.shape[3:]

        if h is None:
            h_shape = (batch_size, self.width, *dom_sizes) if not self.resolution_scaling_factor else (batch_size, self.width,) + tuple([int(round(self.resolution_scaling_factor*s)) for s in dom_sizes])
            h = torch.zeros(h_shape).to(x.device)
            h += self.bias_h

        outputs = []
        for i in range(timesteps):
            h = self.cell(x[:, i], h)
            if self.return_sequences:
                outputs.append(h)

        if self.return_sequences:
            return torch.stack(outputs, dim=1)
        else:
            return h