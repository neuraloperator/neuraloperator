
"""
This solves the 1D (invisid) Burgers equation using FC-PINO after self-similar transformation.

More specifically:

Consider the 1D inviscid Burgers equation:

u_t + u*u_x = 0

If we consider the ansantz:

u = (1-t)^(lambda) * U(x / (1-t)^(1 + lambda))

for a fixed lambda in the positive reals, we get the folliwing equation:

-lambda*U + ((1+lambda)*y + U) * U_y = 0

for spatial variable y.

In the case that lambda = (1/(2i + 2)) for i in the non-negative integers, we get smooth solutions for U.

Solcing this using FC-PINO, setting the Boundary conditions to be:

U(-2) = 1

and notitng that the solution must be odd, we can set

U(2) = -1 and lambda = 1/2

and solve this equation using FC-PINO on domain [-2, 2].

For more detials, see:

https://arxiv.org/pdf/2211.15960
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam


import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator


from neuralop.losses.meta_losses import Relobralo
from neuralop.layers.fourier_continuation import FCLegendre, FCGram
from neuralop.models.fc_fno import FC_FNO


torch.manual_seed(23)
np.random.seed(23)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


plotting = False

## Fourier Continuation function and parameters - can be gram/legendre
## For more details see neuralop.layers.fourier_continuation
contFunc = 'gram'
contPoints = 50
degree = 6

## FNO Specific Parameters
modes = 24
lr = 0.0001


plotInterval = 15000 ## How often you want to plot

nEpochs = 60001 ## How many epochs to train for

useSmoothLoss = True ## Whether to use the smoothness loss

patience = 1000 ## Patience for Scheduler

## Lambda value in burgers equation
l = 0.5 

## Domain of the problem 
## NOTE: For self-simlar burgers equation, domain MUST be symmetric (i.e must go from (-n, n))
resolution1D = 400
domain = (-2, 2) 

if plotting:
    fcName = 'FC-PINO'
    plotting_directory = f'{fcName}_{contFunc}_1D_Burgers_plots'
    os.makedirs(plotting_directory, exist_ok=True)

###############################
# CONTINUATION FUNCTIONS
###############################
if contFunc.lower() == 'gram':
    extension = FCGram(d=degree, n_additional_pts=contPoints).to('cuda')
elif contFunc.lower() == 'legendre':
    extension = FCLegendre(d=degree, n_additional_pts=contPoints).to('cuda')

###############################
# MODEL INITIALIZATION
###############################
model = FC_FNO(
    in_channels=1,
    Lengths= (domain[1] - domain[0],), ## Domain length
    out_channels = 1,  
    n_modes=(modes,),
    hidden_channels = 200,
    n_layers = 4,
    FC_obj=extension,
    non_linearity=F.tanh,
    projection_nonlinearity=F.tanh, ## Non-linearity of the projecton in FC_FNO, must be tanh or silu
).to('cuda')

params = [*model.parameters()]
relobralo = Relobralo(params=params, num_losses=3)

y = torch.linspace(domain[0], domain[1], resolution1D + 1, device='cuda', dtype=torch.float64)[:-1].unsqueeze(0).unsqueeze(0)

def getLosses(model, ep):

    ## FC-PINO, computing derivatives dx and dxx as in Burgers Equation
    U, dxArr = model(y, derivs_to_compute=['dx', 'dxx'])

    ## Extract the derivatives and the solution
    U = U.squeeze() ## (1, 1 x_res) ---> (x_res)
    Uy = dxArr[0].squeeze() ## (1, 1 x_res) ---> (x_res)
    Uyy = dxArr[1].squeeze() ## (1, 1 x_res) ---> (x_res)

    ## Boundary loss
    lossB = torch.norm(U[0] - 1) ** 2 + torch.norm(U[(resolution1D+1)//2]) ** 2 + torch.norm(U[-1] + 1) ** 2 

    ## Interior loss
    intLoss = -l*U + ((1+l)*y + U) * Uy   
    lossI = torch.norm(intLoss)**2 / resolution1D

    # Smoothness loss
    if useSmoothLoss:
        smoothLoss = ((1+l)*y + U) * Uyy + (1 + Uy) * Uy 
        smoothnessLoss = torch.norm(smoothLoss)**2 / resolution1D
    else:
        smoothnessLoss = 0
    
    ## Compute ReLoBraLo loss, For More, see neuralop.losses.meta_losses
    if useSmoothLoss:
        lossDict = {'Interior': lossI, 'Boundary': lossB, 'Smoothness': smoothnessLoss}
        totalLoss, lambdas = relobralo(lossDict, step=ep)
    else:
        lossDict = {'Interior': lossI, 'Boundary': lossB}
        totalLoss, lambdas = relobralo(lossDict, step=ep)
    
    return {
        "interior": lossI,
        "boundary": lossB, 
        "smoothness": smoothnessLoss, 
        "total": totalLoss, 
        "lambdas": lambdas,
        "y": y,
    }

## Set an Optimizer and Scheduler
optimizer = Adam(params, lr=lr, weight_decay=0) 
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.5)


################################################################
# Main training loop
################################################################
for ep in range(nEpochs):

    optimizer.zero_grad()

    losses = getLosses(model, ep)
    loss = losses["total"]

    loss.backward()
    optimizer.step()
    scheduler.step(loss) 

    print(f"epoch = {ep},\tloss = {loss.item()}") 
            
    x = losses["y"]
    
    if plotting == True:
        if ep % plotInterval == 0:
            U, dxArr = model(y, derivs_to_compute=['dx', 'dxx'])
            Uy = dxArr[0].squeeze()
            Uyy = dxArr[1].squeeze()
            U = U.squeeze()
    
            ynp = y.squeeze().cpu().detach().numpy()
            unp  = U.squeeze().detach().cpu().numpy()
            dUnp = Uy.squeeze().detach().cpu().numpy()
            dUUnp = Uyy.squeeze().detach().cpu().numpy()

            ## Combined plot: U, dU, d²U all on one figure
            fcNameTitle = fcName.replace('-', '-').title()
            plt.figure(figsize=(14, 10))
            plt.title(f'{fcNameTitle} FC-{contFunc.capitalize()}', fontsize=38)
            plt.plot(ynp, unp, label=r'$U_\theta(y)$', linewidth=3)
            plt.plot(ynp, dUnp, label=r'$\partial_y (U_\theta(y))$', linewidth=3)
            plt.plot(ynp, dUUnp, label=r'$\partial_{yy} (U_\theta(y))$', linewidth=3)
            plt.xlabel(r'$y$', fontsize=32)
            plt.legend(fontsize=33)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(f"{plotting_directory}/U_dU_dUU_overlay_ep{ep}.pdf", dpi=300)
            plt.close()
