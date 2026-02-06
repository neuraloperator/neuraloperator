"""
Train FC-PINO on the 1D inviscid Burgers equation via self-similar reduction.

Problem setup
------------
The 1D inviscid Burgers equation is:

    u_t + u * u_x = 0

Using the self-similar ansatz:

    u = (1 - t)^lambda * U(y),   y = x / (1 - t)^(1 + lambda)

for lambda > 0, the PDE reduces to an ODE in the spatial variable y:

    -lambda * U + ((1 + lambda) * y + U) * U_y = 0

For lambda = 1 / (2*i + 2) with i a non-negative integer, U admits smooth solutions. 

This script uses lambda = 1/2 and enforces boundary conditions
U(-2) = 1 and U(2) = -1 (odd solution) on the domain y in [-2, 2].

Training is physics-informed: the model predicts U(y), and losses are defined from the residual 
of the ODE, boundary conditions, and optional smoothness (differentiated residual).

Reference
---------
https://arxiv.org/pdf/2211.15960
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from neuralop.losses.meta_losses import Relobralo
from neuralop.layers.fourier_continuation import FCLegendre, FCGram
from neuralop.models.fc_fno import FC_FNO

torch.manual_seed(23)
np.random.seed(23)

# ---------------------------------------------------------------------------
# Options and hyperparameters
# ---------------------------------------------------------------------------

# PDE parameter: lambda in the self-similar reduction (smooth solution for lambda=1/2)
l = 0.5

# Spatial domain for y. Must be symmetric (-a, a) for the self-similar formulation.
resolution1D = 400
domain = (-2, 2)

# Fourier continuation: 'gram' or 'legendre' (see neuralop.layers.fourier_continuation)
fc_choice = "gram"
n_additional_pts = 50  # Number of additional points for continuation
degree = 6  # Degree for continuation
fc_classes = {"gram": FCGram, "legendre": FCLegendre}
FC_object = fc_classes[fc_choice.lower()](d=degree, n_additional_pts=n_additional_pts).to("cuda")

# Training parameters
n_epochs = 60001  # Total training epochs
lr = 0.0001  # Learning rate
patience = 1000  # Epochs without improvement before LR is reduced (scheduler)
useSmoothnessLoss = True  # Include smoothness loss (differentiated residual)


# ---------------------------------------------------------------------------
# FC-FNO model: 1D input U(y), outputs U and optional derivatives
# ---------------------------------------------------------------------------
model = FC_FNO(
    in_channels=1,
    domain_lengths=(domain[1] - domain[0],),
    out_channels=1,
    n_modes=(24,),
    hidden_channels=200,
    n_layers=4,
    FC_object=FC_object,
    non_linearity=F.tanh,
    projection_nonlinearity=F.tanh,  # Must be tanh or silu in FC_FNO
).to("cuda")

params = list(model.parameters())
relobralo = Relobralo(params=params, num_losses=3)  # Multi-loss balancing

# Collocation grid: y in [domain[0], domain[1)], shape (1, 1, resolution1D)
y = torch.linspace(domain[0], domain[1], resolution1D + 1, device="cuda", dtype=torch.float64)
y = y[:-1].unsqueeze(0).unsqueeze(0)


def getLosses(model, ep):
    """
    Compute physics-informed losses for the self-similar Burgers ODE.

    Residual: -lambda*U + ((1+lambda)*y + U)*U_y = 0.
    Boundary: U(-2)=1, U(0)=0 (odd), U(2)=-1.
    Optional smoothness: derivative of residual w.r.t. y.

    Returns
    -------
    dict
        Keys: "interior", "boundary", "smoothness", "total", "lambdas", "y".
    """

    # Forward pass: get U and first/second derivatives w.r.t. y
    U, dxArr = model(y, derivs_to_compute=["dx", "dxx"])

    U = U.squeeze()  # (1, 1, n) -> (n)
    Uy = dxArr[0].squeeze()  # dU/dy
    Uyy = dxArr[1].squeeze()  # d²U/dy²

    # Boundary loss: U(left)=1, U(mid)=0 (odd), U(right)=-1
    lossB = (
        torch.norm(U[0] - 1) ** 2
        + torch.norm(U[(resolution1D + 1) // 2]) ** 2
        + torch.norm(U[-1] + 1) ** 2
    )

    # Interior (PDE) loss: residual of -lambda*U + ((1+lambda)*y + U)*U_y = 0
    residual = -l * U + ((1 + l) * y.squeeze() + U) * Uy
    lossI = torch.norm(residual) ** 2 / resolution1D

    # Smoothness loss: d/dy of residual (encourages smoother solutions)
    if useSmoothnessLoss:
        smoothness_residual = ((1 + l) * y.squeeze() + U) * Uyy + (1 + Uy) * Uy
        smoothnessLoss = torch.norm(smoothness_residual) ** 2 / resolution1D
        lossDict = {"Interior": lossI, "Boundary": lossB, "Smoothness": smoothnessLoss}
    else:
        smoothnessLoss = torch.tensor(0.0, device=U.device, dtype=U.dtype)
        lossDict = {"Interior": lossI, "Boundary": lossB}
    totalLoss, lambdas = relobralo(lossDict, step=ep)

    return {
        "interior": lossI,
        "boundary": lossB,
        "smoothness": smoothnessLoss,
        "total": totalLoss,
        "lambdas": lambdas,
        "y": y,
    }


# Optimizer and learning-rate scheduler
optimizer = optim.AdamW(params, lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.5)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
for ep in range(n_epochs):
    optimizer.zero_grad()
    losses = getLosses(model, ep)
    loss = losses["total"]
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    if ep % 100 == 0 or ep == n_epochs - 1:
        print(f"epoch = {ep},\tloss = {loss.item()}")
