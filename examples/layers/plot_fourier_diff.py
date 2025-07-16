"""
.. _fourier_diff :

Fourier Differentiation
========================================================
An example of usage of our Fourier Differentiation Function on 1d data.
"""

# %%
# Import the library
# ------------------
# We first import our `neuralop` library and required dependencies.
import torch
import numpy as np
import matplotlib.pyplot as plt
from neuralop.losses.fourier_diff import fourier_derivative_1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# %%
# Creating an example of periodic 1D curve
# --------------------
# Here we consider sin(x) and cos(x), which are periodic on the interval [0,2pi]
L = 2*torch.pi
x = torch.linspace(0, L, 101)[:-1]
f = torch.stack([torch.sin(x), torch.cos(x)], dim=0)
x_np = x.cpu().numpy()

# %%
# Differentiate the signal
# -----------------------------------------
# We use the Fourier differentiation function to differentiate the signal
dfdx = fourier_derivative_1d(f, order=1, L=L)
df2dx2 = fourier_derivative_1d(f, order=2, L=L)
df3dx3 = fourier_derivative_1d(f, order=3, L=L)


# %%
# Plot the results for sin(x)
# ----------------------
plt.figure()
plt.plot(x_np, dfdx[0].squeeze().cpu().numpy(), label='Fourier dfdx')
plt.plot(x_np, np.cos(x_np), '--', label='dfdx')
plt.plot(x_np, df2dx2[0].squeeze().cpu().numpy(), label='Fourier df2dx2')
plt.plot(x_np, -np.sin(x_np), '--', label='df2dx2')
plt.plot(x_np, df3dx3[0].squeeze().cpu().numpy(), label='Fourier df3dx3')
plt.plot(x_np, -np.cos(x_np), '--', label='df3dx3')
plt.xlabel('x')
plt.legend()
plt.show()

# %%
# Plot the results for cos(x)
# ----------------------
plt.figure()
plt.plot(x_np, dfdx[1].squeeze().cpu().numpy(), label='Fourier dfdx')
plt.plot(x_np, -np.sin(x_np), '--', label='dfdx')
plt.plot(x_np, df2dx2[1].squeeze().cpu().numpy(), label='Fourier df2dx2')
plt.plot(x_np, -np.cos(x_np), '--', label='df2dx2')
plt.plot(x_np, df3dx3[1].squeeze().cpu().numpy(), label='Fourier df3dx3')
plt.plot(x_np, np.sin(x_np), '--', label='df3dx3')
plt.xlabel('x')
plt.legend()
plt.show()
        
        
        
# %%
# Creating an example of non-periodic 1D curve
# --------------------
# Here we consider sin(16x)-cos(8x) and exp(-0.8x)+sin(x)
L = 2*torch.pi
x = torch.linspace(0, L, 101)[:-1]    
f = torch.stack([torch.sin(3*x) - torch.cos(x), torch.exp(-0.8*x)+torch.sin(x)], dim=0)
x_np = x.cpu().numpy()

# %%
# Differentiate the signal
# -----------------------------------------
# We use the Fourier differentiation function with Fourier continuation to differentiate the signal
dfdx = fourier_derivative_1d(f, order=1, L=L, use_FC='Legendre', FC_d=4, FC_n_additional_pts=30, FC_one_sided=False)
df2dx2 = fourier_derivative_1d(f, order=2, L=L, use_FC='Legendre', FC_d=4, FC_n_additional_pts=30, FC_one_sided=False)

     
# %%
# Plot the results for sin(16x)-cos(8x)
# ----------------------
plt.figure()
plt.plot(x_np, dfdx[0].squeeze().cpu().numpy(), label='Fourier dfdx')
plt.plot(x_np, 3*torch.cos(3*x) + torch.sin(x), '--', label='dfdx')
plt.plot(x_np, df2dx2[0].squeeze().cpu().numpy(), label='Fourier df2dx2')
plt.plot(x_np, -9*torch.sin(3*x) + torch.cos(x), '--', label='df2dx2')
plt.xlabel('x')
plt.legend()
plt.show()

# %%
# Plot the results for exp(-0.8x)+sin(x)
# ----------------------
plt.figure()
plt.plot(x_np, dfdx[1].squeeze().cpu().numpy(), label='Fourier dfdx')
plt.plot(x_np, -0.8*torch.exp(-0.8*x)+torch.cos(x), '--', label='dfdx')
plt.plot(x_np, df2dx2[1].squeeze().cpu().numpy(), label='Fourier df2dx2')
plt.plot(x_np, 0.64*torch.exp(-0.8*x)-torch.sin(x), '--', label='df2dx2')
plt.xlabel('x')
plt.legend()
plt.show()