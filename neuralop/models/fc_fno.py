from functools import partialmethod
from typing import Tuple, List, Union, Literal
from functools import partial

Number = Union[float, int]

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.embeddings import GridEmbeddingND, GridEmbedding2D
from ..layers.spectral_convolution import SpectralConv
from ..layers.fno_block import FNOBlocks
from ..layers.channel_mlp import ChannelMLP, LinearChannelMLP
from ..layers.complex import ComplexValued
from ..models.base_model import BaseModel
from ..models.fno import FNO
from ..losses.differentiation import *
from ..layers.fourier_continuation import FCGram, FCLegendre

torch.set_default_dtype(torch.float64)


class FC_FNO(FNO):
    """
    Fourier Continuation Neural Operator

    The architecture is described in [1]_.

    FC_FNO integrates Fourier Continuation (FC) into operator learning for non-periodic PDE.

    Parameters
    ----------
    All arguments from :class:`neuralop.models.FNO` are also avaliable here, with the exception of:
    - domain padding,
    - complex_data,
    - output_shape,
    - resolution_scaling_factor.

    Additional arguments:
    ---------------------
    FC_obj: Instantiated Fourier Continuation object. See :class:`neuralop.layers.fourier_continuation`
    projection_nonlinearity: Nonlinearity of the projection. Must be tanh or sigmoid.
    Lengths: Lengths of the input domain. Must be a tuple

    Notes
    -----
    - Input data must be of type torch.float64.
    - When specifying derivs_to_compute, input a list of strings containing 
    'dx', 'dxx', 'dy', 'dyy', 'dz', 'dzz'. x represents the first dimension, y the second, and z the third.
    - Derivatives are computed using the chain rule, and as implemented, work up to 2nd-Order.

    See the FNO documentation for detailed descriptions.

    References
    ----------
    .. [1] Ganeshram, A., Maust H., Duruisseaux V., Li, Z., Wang Y., Leibovici D., Bruno O., Hou T., & Anandkumar, A.
        "FC-PINO: High Precision Physics-Informed Neural Operators via
            Fourier Continuation" (2025). arXiv preprint arXiv:2211.15960.
        https://arxiv.org/pdf/2211.15960
    """

    _dQ_DOCSTRING = """
        Compute the model's derivatives via the chain rule.

        More specifically:
        --------
        1) We first compute spectral derivatives before the final projection. Denote these as dV/dx, where
        V is the pre-projection feature field and x is the input grid.
        2) We then use projection-layer derivatives (e.g., dQ/dV) and the chain tule to 
        compute the derivatives of the final model output with respect to input x.
        3) We use `einsum` to perform the required matrix multiplications efficiently.

        Notes
        -----
        - Derivatives are up to second order
        - Assumes the final nonlinearity is `tanh` or `sigmoid`

        Chain Rule Identities
        ---------------------
        Gradient:
            D(f ∘ g)(x) = Df(g(x)) · Dg(x)

        Hessian (2nd derevative):
            D²(f ∘ g)(x) = D²f(g(x))[Dg(x), Dg(x)] + Df(g(x)) · D²g(x)

        Higher-order:
            Dⁿ(f ∘ g)(x) = Σ_{k=0}^{n-1} C(n-1, k) · D^{k+1}g(x) · D^{n-k}(f'(g(x)))

        The same approach generalizes to higher dimensions and derivative orders by
        applying the corresponding multivariate chain rule. These idenities follow easily from
        the chain and product rule. 

        """

    def __init__(
        self,
        n_modes: Tuple[int],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Literal["ada_in", "group_norm", "instance_norm"] = None,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating"] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating"] = "linear",
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = dict(),
        separable: bool = False,
        preactivation: bool = False,
        conv_module: nn.Module = SpectralConv,
        FC_obj=None,
        projection_nonlinearity=F.tanh,
        Lengths=Tuple[float],
    ):

        super().__init__(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            lifting_channel_ratio=lifting_channel_ratio,
            projection_channel_ratio=projection_channel_ratio,
            positional_embedding=positional_embedding,
            non_linearity=non_linearity,
            norm=norm,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
            fno_skip=fno_skip,
            fno_block_precision=fno_block_precision,
            stabilizer=stabilizer,
            max_n_modes=max_n_modes,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            separable=separable,
            preactivation=preactivation,
            conv_module=conv_module,
        )

        ## Convert FNO block weights to complex double precision for higher precision
        for i in range(n_layers):
            self.fno_blocks.convs[i].weight = self.fno_blocks.convs[i].weight.to(torch.cdouble)

        self.FC_obj = FC_obj
        self.projection_nonlinearity = projection_nonlinearity
        self.Lengths = Lengths

        ## Linear projection MLP
        self.projection = LinearChannelMLP(
            layers=[hidden_channels, self.projection_channels, out_channels],
            non_linearity=self.projection_nonlinearity,
        )


        type(self).dQ_1D.__doc__ = self._dQ_DOCSTRING
        type(self).dQ_2D.__doc__ = self._dQ_DOCSTRING
        type(self).dQ_3D.__doc__ = self._dQ_DOCSTRING

    def dQ_1D(self, X1, Dx_arr, Q1, Q2, derivs_to_compute):

        # ----------------------------
        # dQ_1D EINSUM KEY
        # ----------------------------

        # Indices:
        #   b : batch
        #   i : input/feature channel index (hidden_channels before projection)
        #   m : intermediate channel index (projection hidden / Q1 out)
        #   o : output channel index (out_channels after projection)
        #   x : spatial grid index
        #
        # Tensors / shapes (consistent with permutes):
        #   X1   : (b, m, x)          [after X1 = X1.permute(0, 2, 1)]
        #   dP1  : (b, m, x)          a'(X1), where a represents the non-linear activation functuon
        #   dP2  : (b, m, x)          a''(X1)
        #   dW1  : (m, i)             Q1.weight
        #   dW2  : (o, m)             Q2.weight   (as used in dQ_1D)
        #   dx   : (b, i, x)          dx[V]
        #   dxx  : (b, i, x)          dxx[V]
        
        # ----------------------------
        # Einsums Used:
        #   dQ = einsum("mi,bmx,om->boix")               => (b, o, i, x)
        #       -> Jacobian wrt input features:  Q2 · diag(a'(X1)) · Q1
        #   Dx = einsum("boix,bix->box")                 => (b, o, x)
        #       -> Apply Jacobian to dx[V]:  (dQ)·dx
        #   W = einsum("mi,bix->bmx")                   => (b, m, x)
        #       -> Push dx[V] through Q1:  Q1·dx
        #   H2 = einsum("om,bmx->bomx")                  => (b, o, m, x)
        #       -> Q2-weighted a'' term:  Q2 ⊙ a''(X1)
        #   D2X1 = einsum("bomx,bmx,bmx->box")             => (b, o, x)
        #       -> a'' contribution:  Q2·(a'' ⊙ (Q1·dx)^2)
        #   D2X2 = einsum("boix,bix->box")                 => (b, o, x)
        #       -> a'·dxx contribution:  (dQ)·dxx
        # ----------------------------

        Dx_out = []

        need_dx = "dx" in derivs_to_compute or "dxx" in derivs_to_compute
        if need_dx:
            dx = Dx_arr["dx"]
        if "dxx" in derivs_to_compute:
            dxx = Dx_arr["dxx"]

        X1 = X1.permute(0, 2, 1)

        dW1 = Q1.weight
        dW2 = Q2.weight

        # Exact deriv of activation function
        if self.projection_nonlinearity == F.tanh:
            dP1 = 1 / torch.cosh(X1) ** 2
        elif self.projection_nonlinearity == F.silu:
            dP1 = torch.sigmoid(X1) * (1 + X1 * (1 - torch.sigmoid(X1)))
        else:
            raise ValueError(
                f"Projection nonlinearity {self.projection_nonlinearity} not supported. Must be F.tanh or F.silu"
            )

        dQ = torch.einsum("mi,bmx,om->boix", dW1, dP1, dW2)

        if "dx" in derivs_to_compute:
            Dx = torch.einsum("boix,bix->box", dQ, dx)
            Dx_out.append(Dx)

        if self.projection_nonlinearity == F.tanh:
            dP2 = -2 * dP1 * torch.tanh(X1)
        elif self.projection_nonlinearity == F.silu:
            dP2 = (
                torch.sigmoid(X1)
                * (1 - torch.sigmoid(X1))
                * (2 + X1 * (1 - 2 * torch.sigmoid(X1)))
            )
        else:
            raise ValueError(
                f"Projection nonlinearity {self.projection_nonlinearity} not supported. Must be F.tanh or F.silu"
            )

        if "dxx" in derivs_to_compute:

            W = torch.einsum("mi,bix->bmx", dW1, dx)  
            H2 = torch.einsum("om,bmx->bomx", dW2, dP2)  

            d2X_1 = torch.einsum("bomx,bmx,bmx->box", H2, W, W)
            d2X_2 = torch.einsum("boix,bix->box", dQ, dxx)

            dxx = d2X_1 + d2X_2
            Dx_out.append(dxx)

        return Dx_out

    def dQ_2D(self, X1, Dx_arr, Q1, Q2, derivs_to_compute):

        # ----------------------------
        # dQ_2D EINSUM KEY
        # ----------------------------
        # Indices:
        #   b : batch
        #   i : input/feature channel index (hidden_channels before projection)
        #   m : intermediate channel index (projection hidden / Q1 out)
        #   o : output channel index (out_channels after projection)
        #   x : first grid axis 
        #   t : second grid axis
        #
        # Tensors / shapes (consistent with permutes):
        #   X1   : (b, m, x, t)       [after X1 = X1.permute(0, 3, 1, 2)]
        #   dP1  : (b, m, x, t)       a'(X1), where a represents the non-linear activation function
        #   dP2  : (b, m, x, t)       a''(X1)
        #   dW1  : (m, i)             Q1.weight
        #   dW2  : (m, o)             Q2.weight.t()
        #   wx   : (b, i, t, x)       dx[V]  (as provided in Dx_arr)
        #   wy   : (b, i, t, x)       dy[V]  (as provided in Dx_arr)
        #   wxx  : (b, i, t, x)       dxx[V]
        #   wyy  : (b, i, t, x)       dyy[V]
        
        # ----------------------------
        # Einsums used:
        #   dQ   = einsum("mi,bmtx,mo->boitx")             => (b, o, i, t, x)
        #       -> Jacobian wrt input features:  Q2 · diag(a'(X1)) · Q1
        #   wxQ  = einsum("boitx,bitx->botx")              => (b, o, t, x)
        #       -> Apply Jacobian to dx[V]:  (dQ)·dx
        #   wyQ  = einsum("boitx,bitx->botx")              => (b, o, t, x)
        #       -> Apply Jacobian to dy[V]:  (dQ)·dy
        #   H2   = einsum("mo,bmtx->bomtx")                => (b, o, m, t, x)
        #       -> Q2-weighted a'' term:  Q2 ⊙ a''(X1)
        #   wxx1 = einsum("bitx,mi,bomtx,mj,bjtx->botx")   => (b, o, t, x)
        #       -> a'' contribution:  Q2·(a'' ⊙ (Q1·dx)^2)
        #   wxx2 = einsum("boitx,bitx->botx")              => (b, o, t, x)
        #       -> a'·dxx contribution:  (dQ)·dxx
        #   wyy1 = einsum("bitx,mi,bomtx,mj,bjtx->botx")   => (b, o, t, x)
        #       -> a'' contribution:  Q2·(a'' ⊙ (Q1·dy)^2)
        #   wyy2 = einsum("boitx,bitx->botx")              => (b, o, t, x)
        #       -> a'·dyy contribution:  (dQ)·dyy
        # ----------------------------

        Dx_out = []
        need_dx = "dx" in derivs_to_compute or "dxx" in derivs_to_compute
        need_dy = "dy" in derivs_to_compute or "dyy" in derivs_to_compute
        if need_dx:
            wx = Dx_arr["dx"]
        if need_dy:
            wy = Dx_arr["dy"]

        X1 = X1.permute(0, 3, 1, 2)

        b = X1.shape[0]
        x = X1.shape[2]
        t = X1.shape[3]
        i = self.hidden_channels
        m = Q1.weight.shape[0]

        dW1 = Q1.weight

        # Derivative of activation
        if self.projection_nonlinearity == F.tanh:
            dP1 = 1 / torch.cosh(X1) ** 2
        elif self.projection_nonlinearity == F.silu:
            dP1 = torch.sigmoid(X1) * (1 + X1 * (1 - torch.sigmoid(X1)))
        else:
            raise ValueError(
                f"Projection nonlinearity {self.projection_nonlinearity} not supported. Must be F.tanh or F.silu"
            )

        dW2 = Q2.weight.t()

        dQ = torch.einsum("mi,bmtx,mo->boitx", dW1, dP1, dW2)

        if "dx" in derivs_to_compute:
            wxQ = torch.einsum("boitx,bitx->botx", dQ, wx)
            Dx_out.append(wxQ)

        if "dy" in derivs_to_compute:
            wyQ = torch.einsum("boitx,bitx->botx", dQ, wy)
            Dx_out.append(wyQ)

        if "dxx" in derivs_to_compute:
            wxx = Dx_arr["dxx"]
        if "dyy" in derivs_to_compute:
            wyy = Dx_arr["dyy"]

        if self.projection_nonlinearity == F.tanh:
            dP2 = -2 * dP1 * torch.tanh(X1)
        elif self.projection_nonlinearity == F.silu:
            dP2 = (
                torch.sigmoid(X1)
                * (1 - torch.sigmoid(X1))
                * (2 + X1 * (1 - 2 * torch.sigmoid(X1)))
            )
        else:
            raise ValueError(
                f"Projection nonlinearity {self.projection_nonlinearity} not supported. Must be F.tanh or F.silu"
            )

        H2 = torch.einsum("mo,bmtx->bomtx", dW2, dP2)

        if "dxx" in derivs_to_compute:
            wxx1 = torch.einsum("bitx,mi,bomtx,mj,bjtx->botx", wx, dW1, H2, dW1, wx)
            wxx2 = torch.einsum("boitx,bitx->botx", dQ, wxx)
            wxxQ = wxx1 + wxx2
            Dx_out.append(wxxQ)

        if "dyy" in derivs_to_compute:
            wyy1 = torch.einsum("bitx,mi,bomtx,mj,bjtx->botx", wy, dW1, H2, dW1, wy)
            wyy2 = torch.einsum("boitx,bitx->botx", dQ, wyy)
            wyyQ = wyy1 + wyy2
            Dx_out.append(wyyQ)

        return Dx_out

    def dQ_3D(self, X1, Dx_arr, Q1, Q2, derivs_to_compute):

        # ----------------------------
        # dQ_3D EINSUM KEY
        # ----------------------------

        # Indices:
        #   b : batch
        #   i : input/feature channel index (hidden_channels before projection)
        #   c : intermediate channel index (projection hidden / Q1 out)
        #   o : output channel index (out_channels after projection)
        #   t : grid axis 1
        #   x : grid axis 2
        #   z : grid axis 3
    
        # Tensors / shapes (consistent with permutes):
        #   X1   : (b, c, t, x, z)    [after X1 = X1.permute(0, 4, 1, 2, 3)]
        #   dP1  : (b, c, t, x, z)    a'(X1)
        #   dP2  : (b, c, t, x, z)    a''(X1)
        #   dW1  : (c, i)             Q1.weight
        #   dW2  : (c, o)             Q2.weight.t()
        #   dx   : (b, i, t, x, z)    dx[V]
        #   dy   : (b, i, t, x, z)    dy[V]
        #   dz   : (b, i, t, x, z)    dz[V]
        #   dxx  : (b, i, t, x, z)    dxx[V]
        #   dyy  : (b, i, t, x, z)    dyy[V]
        #   dzz  : (b, i, t, x, z)    dzz[V]

        # ----------------------------
        # dQ_3D Einsums used:
        #   dQ   = einsum("ci,bctxz,co->boitxz")                 => (b, o, i, t, x, z)
        #       -> Jacobian wrt input features:  Q2 · diag(a'(X1)) · Q1
        #   wxQ  = einsum("boitxz,bitxz->botxz")                 => (b, o, t, x, z)
        #       -> Apply Jacobian to dx[V]:  (dQ)·dx
        #   wyQ  = einsum("boitxz,bitxz->botxz")                 => (b, o, t, x, z)
        #       -> Apply Jacobian to dy[V]:  (dQ)·dy
        #   wzQ  = einsum("boitxz,bitxz->botxz")                 => (b, o, t, x, z)
        #       -> Apply Jacobian to dz[V]:  (dQ)·dz
        #   H2   = einsum("co,bctxz->bcotxz")                    => (b, c, o, t, x, z)
        #       -> Q2-weighted a'' term:  Q2 ⊙ a''(X1)
        #   wxx1 = einsum("bitxz,ci,bcotxz,cj,bjtxz->botxz")     => (b, o, t, x, z)
        #       -> a'' contribution:  Q2·(a'' ⊙ (Q1·dx)^2)
        #   wxx2 = einsum("boitxz,bitxz->botxz")                 => (b, o, t, x, z)
        #       -> a'·dxx contribution:  (dQ)·dxx
        #   wyy1 = einsum("bitxz,ci,bcotxz,cj,bjtxz->botxz")     => (b, o, t, x, z)
        #       -> a'' contribution:  Q2·(a'' ⊙ (Q1·dy)^2)
        #   wyy2 = einsum("boitxz,bitxz->botxz")                 => (b, o, t, x, z)
        #       -> a'·dyy contribution:  (dQ)·dyy
        #   wzz1 = einsum("bitxz,ci,bcotxz,cj,bjtxz->botxz")     => (b, o, t, x, z)
        #       -> a'' contribution:  Q2·(a'' ⊙ (Q1·dz)^2)
        #   wzz2 = einsum("boitxz,bitxz->botxz")                 => (b, o, t, x, z)
        #       -> a'·dzz contribution:  (dQ)·dzz
        # ----------------------------

        Dx_out = []

        need_dx = "dx" in derivs_to_compute or "dxx" in derivs_to_compute
        need_dy = "dy" in derivs_to_compute or "dyy" in derivs_to_compute
        need_dz = "dz" in derivs_to_compute or "dzz" in derivs_to_compute
       
        dx = Dx_arr["dx"] if need_dx else None
        dy = Dx_arr["dy"] if need_dy else None
        dz = Dx_arr["dz"] if need_dz else None

        dxx = Dx_arr["dxx"] if "dxx" in derivs_to_compute else None
        dyy = Dx_arr["dyy"] if "dyy" in derivs_to_compute else None
        dzz = Dx_arr["dzz"] if "dzz" in derivs_to_compute else None

        X1 = X1.permute(0, 4, 1, 2, 3)

        B, C, T, X, Z = X1.shape
        I = self.hidden_channels
        O = self.out_channels

        dW1 = Q1.weight
        dW2 = Q2.weight.t()

        if self.projection_nonlinearity == F.tanh:
            dP1 = 1 / torch.cosh(X1) ** 2  # (B, C, T, X, Z)
        elif self.projection_nonlinearity == F.silu:
            dP1 = torch.sigmoid(X1) * (1 + X1 * (1 - torch.sigmoid(X1)))
        else:
            raise ValueError(
                f"Projection nonlinearity {self.projection_nonlinearity} not supported. Must be F.tanh or F.silu"
            )

        dQ = torch.einsum("ci, bctxz, co -> boitxz", dW1, dP1, dW2)

        if "dx" in derivs_to_compute:
            wxQ = torch.einsum("boitxz,bitxz->botxz", dQ, dx)
            Dx_out.append(wxQ)

        if "dy" in derivs_to_compute:
            wyQ = torch.einsum("boitxz,bitxz->botxz", dQ, dy)
            Dx_out.append(wyQ)

        if "dz" in derivs_to_compute:
            wzQ = torch.einsum("boitxz,bitxz->botxz", dQ, dz)
            Dx_out.append(wzQ)

        if self.projection_nonlinearity == F.tanh:
            dP2 = -2 * dP1 * torch.tanh(X1)
        elif self.projection_nonlinearity == F.silu:
            dP2 = (
                torch.sigmoid(X1)
                * (1 - torch.sigmoid(X1))
                * (2 + X1 * (1 - 2 * torch.sigmoid(X1)))
            )
        else:
            raise ValueError(
                f"Projection nonlinearity {self.projection_nonlinearity} not supported. Must be F.tanh or F.silu"
            )

        H2 = torch.einsum("co,bctxz->bcotxz", dW2, dP2) 

        if "dxx" in derivs_to_compute:
            wxx1 = torch.einsum("bitxz,ci,bcotxz,cj,bjtxz->botxz", dx, dW1, H2, dW1, dx)
            wxx2 = torch.einsum("boitxz,bitxz->botxz", dQ, dxx)
            wxxQ = wxx1 + wxx2
            Dx_out.append(wxxQ)
        if "dyy" in derivs_to_compute:
            wyy1 = torch.einsum("bitxz,ci,bcotxz,cj,bjtxz->botxz", dy, dW1, H2, dW1, dy)
            wyy2 = torch.einsum("boitxz,bitxz->botxz", dQ, dyy)
            wyyQ = wyy1 + wyy2
            Dx_out.append(wyyQ)
        if "dzz" in derivs_to_compute:
            wzz1 = torch.einsum("bitxz,ci,bcotxz,cj,bjtxz->botxz", dz, dW1, H2, dW1, dz)
            wzz2 = torch.einsum("boitxz,bitxz->botxz", dQ, dzz)
            wzzQ = wzz1 + wzz2
            Dx_out.append(wzzQ)

        return Dx_out

    ## say if we want derivs to compute in the forward
    def forward(self, x, derivs_to_compute = None, output_shape=None, **kwargs):
       
        """FC_FNO's forward pass"""

        if (x.dtype != torch.float64):
            raise ValueError("Input must be of type torch.float64")

        if derivs_to_compute is not None:
            if not isinstance(derivs_to_compute, (list, tuple, set)):
                raise ValueError("derivs_to_compute must be a list of strings")

            derivs_to_compute = [d.lower() for d in list(derivs_to_compute)]
            
            if not derivs_to_compute:
                derivs_to_compute = None
            else:
                if self.n_dim == 1:
                    allowed_derivatives = {"dx", "dxx"}
                    invalid_derivs = set(derivs_to_compute) - allowed_derivatives
                    if invalid_derivs:
                        raise ValueError(
                            f"Unsupported derivatives requested: {sorted(invalid_derivs)}. "
                            f"Allowed for 1D FC_FNO: {sorted(allowed_derivatives)}."
                        )
                elif self.n_dim == 2:
                    allowed_derivatives = {"dx", "dxx", "dy", "dyy"}
                    invalid_derivs = set(derivs_to_compute) - allowed_derivatives
                    if invalid_derivs:
                        raise ValueError(
                            f"Unsupported derivatives requested: {sorted(invalid_derivs)}. "
                            f"Allowed for 2D FC_FNO: {sorted(allowed_derivatives)}."
                        )
                elif self.n_dim == 3:
                    allowed_derivatives = {"dx", "dxx", "dy", "dyy", "dz", "dzz"}
                    invalid_derivs = set(derivs_to_compute) - allowed_derivatives
                    if invalid_derivs:
                        raise ValueError(
                            f"Unsupported derivatives requested: {sorted(invalid_derivs)}. "
                            f"Allowed for 3D FC_FNO: {sorted(allowed_derivatives)}."
                        )
                else:
                    raise ValueError(f"Unsupported dimension: {self.n_dim}. Expected 1, 2, or 3.")
        
        output_shape = [None] * self.n_layers

        # append spatial pos embedding if set
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        if self.n_dim == 1:
            __, __, x_res = x.shape
            x = self.FC_obj(x, dim=1)
        elif self.n_dim == 2:
            __, __, x_res, y_res = x.shape
            x = self.FC_obj(x, dim=2)
        elif self.n_dim == 3:
            __, __, x_res, y_res, z_res = x.shape
            x = self.FC_obj(x, dim=3)
        else:
            raise ValueError(f"Error: expected 1, 2, or 3 dimensions, got {self.n_dim}")

        x = self.lifting(x)

        for layer_idx in range(self.n_layers):
            assert output_shape[layer_idx] is None, "Output shape must be None for FC_FNO"
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])
    

        ## Compute derivatives if end or start
        if derivs_to_compute is not None:
            Dx_arr = {}
            deriv_tuples = []
            if self.n_dim == 1:
                new_Lengths = (
                    self.Lengths[0] * (x_res + self.FC_obj.n_additional_pts) / x_res
                )
                FourierDiff1d = FourierDiff(dim=self.n_dim, L=new_Lengths)
                dx_tuple = 1
                if "dx" in derivs_to_compute:
                    deriv_tuples.append(dx_tuple)
                if "dxx" in derivs_to_compute:
                    dxx_tuple = 2
                    deriv_tuples.append(dxx_tuple)
                if "dxx" in derivs_to_compute and "dx" not in derivs_to_compute:
                    deriv_tuples.append(dx_tuple)
                deriv_array = FourierDiff1d.compute_multiple_derivatives(x, derivatives=deriv_tuples)
                for i, deriv_tuple in enumerate(deriv_tuples):
                    if deriv_tuple == (1):
                        Dx_arr["dx"] = self.FC_obj.restrict(deriv_array[i], 1)
                    elif deriv_tuple == (2):
                        Dx_arr["dxx"] = self.FC_obj.restrict(deriv_array[i], 1)

            elif self.n_dim == 2:
                new_Lengths = (
                    self.Lengths[0] * (x_res + self.FC_obj.n_additional_pts) / x_res,
                    self.Lengths[1] * (y_res + self.FC_obj.n_additional_pts) / y_res,
                )
                FourierDiff2d = FourierDiff(dim=self.n_dim, L=new_Lengths)
                dy_tuple = (0, 1)
                dx_tuple = (1, 0)
                if "dy" in derivs_to_compute:
                    deriv_tuples.append(dy_tuple)
                if "dx" in derivs_to_compute:
                    deriv_tuples.append(dx_tuple)
                if "dyy" in derivs_to_compute:
                    dyy_tuple = (0, 2)
                    deriv_tuples.append(dyy_tuple)
                if "dxx" in derivs_to_compute:
                    dxx_tuple = (2, 0)
                    deriv_tuples.append(dxx_tuple)
                if "dxx" in derivs_to_compute and "dx" not in derivs_to_compute:
                    deriv_tuples.append(dx_tuple)
                if "dyy" in derivs_to_compute and "dy" not in derivs_to_compute:
                    deriv_tuples.append(dy_tuple)
                deriv_array = FourierDiff2d.compute_multiple_derivatives(x, derivatives=deriv_tuples)
                for i, deriv_tuple in enumerate(deriv_tuples):
                    if deriv_tuple == (0, 1):
                        Dx_arr["dy"] = self.FC_obj.restrict(deriv_array[i], 2)
                    elif deriv_tuple == (1, 0):
                        Dx_arr["dx"] = self.FC_obj.restrict(deriv_array[i], 2)
                    elif deriv_tuple == (0, 2):
                        Dx_arr["dyy"] = self.FC_obj.restrict(deriv_array[i], 2)
                    elif deriv_tuple == (2, 0):
                        Dx_arr["dxx"] = self.FC_obj.restrict(deriv_array[i], 2)

            elif self.n_dim == 3:
                new_Lengths = (
                    self.Lengths[0] * (x_res + self.FC_obj.n_additional_pts) / x_res,
                    self.Lengths[1] * (y_res + self.FC_obj.n_additional_pts) / y_res,
                    self.Lengths[2] * (z_res + self.FC_obj.n_additional_pts) / z_res,
                )
                FourierDiff3d = FourierDiff(dim=self.n_dim, L=new_Lengths)
                dz_tuple = (0, 0, 1)
                dy_tuple = (0, 1, 0)
                dx_tuple = (1, 0, 0)
                if "dz" in derivs_to_compute:
                    deriv_tuples.append(dz_tuple)
                if "dy" in derivs_to_compute:
                    deriv_tuples.append(dy_tuple)
                if "dx" in derivs_to_compute:
                    deriv_tuples.append(dx_tuple)
                if "dxx" in derivs_to_compute:
                    dxx_tuple = (2, 0, 0)
                    deriv_tuples.append(dxx_tuple)
                if "dyy" in derivs_to_compute:
                    dyy_tuple = (0, 2, 0)
                    deriv_tuples.append(dyy_tuple)
                if "dzz" in derivs_to_compute:
                    dzz_tuple = (0, 0, 2)
                    deriv_tuples.append(dzz_tuple)
                if "dxx" in derivs_to_compute and "dx" not in derivs_to_compute:
                    deriv_tuples.append(dx_tuple)
                if "dyy" in derivs_to_compute and "dy" not in derivs_to_compute:
                    deriv_tuples.append(dy_tuple)
                if "dzz" in derivs_to_compute and "dz" not in derivs_to_compute:
                    deriv_tuples.append(dz_tuple)
                deriv_array = FourierDiff3d.compute_multiple_derivatives(x, derivatives = deriv_tuples)
                for i, deriv_tuple in enumerate(deriv_tuples):
                    if deriv_tuple == (0, 0, 1):
                        Dx_arr["dz"] = self.FC_obj.restrict(deriv_array[i], 3)
                    elif deriv_tuple == (0, 1, 0):
                        Dx_arr["dy"] = self.FC_obj.restrict(deriv_array[i], 3)
                    elif deriv_tuple == (1, 0, 0):
                        Dx_arr["dx"] = self.FC_obj.restrict(deriv_array[i], 3)
                    elif deriv_tuple == (2, 0, 0):
                        Dx_arr["dxx"] = self.FC_obj.restrict(deriv_array[i], 3)
                    elif deriv_tuple == (0, 2, 0):
                        Dx_arr["dyy"] = self.FC_obj.restrict(deriv_array[i], 3)
                    elif deriv_tuple == (0, 0, 2):
                        Dx_arr["dzz"] = self.FC_obj.restrict(deriv_array[i], 3)

            Q1 = self.projection.fcs[0]  # first Linear layer
            Q2 = self.projection.fcs[-1] # last Linear layer

            ## Restrict the extended input if start
            x = self.FC_obj.restrict(x, self.n_dim)

            ## Compute the derivatives w.r.t. the input
            if self.n_dim == 1:
                X1 = Q1(x.transpose(1, 2))
                Dx_arr = self.dQ_1D(X1, Dx_arr, Q1, Q2, derivs_to_compute)
            elif self.n_dim == 2:
                X1 = Q1(x.permute(0, 2, 3, 1))
                Dx_arr = self.dQ_2D(X1, Dx_arr, Q1, Q2, derivs_to_compute)
            elif self.n_dim == 3:
                X1 = Q1(x.permute(0, 2, 3, 4, 1))
                Dx_arr = self.dQ_3D(X1, Dx_arr, Q1, Q2, derivs_to_compute)
            else:
                raise ValueError(
                    f"Error: expected 1, 2, or 3 dimensions, got {self.n_dim}"
                )

            if self.n_dim == 1:
                x = self.projection(x.transpose(1, 2))
                x = x.transpose(1, 2)
            elif self.n_dim == 2:
                x = self.projection(x.permute(0, 2, 3, 1))
                x = x.permute(0, 3, 1, 2)
            elif self.n_dim == 3:
                x = self.projection(x.permute(0, 2, 3, 4, 1))
                x = x.permute(0, 4, 1, 2, 3)

            return x, Dx_arr
        else:
            return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes