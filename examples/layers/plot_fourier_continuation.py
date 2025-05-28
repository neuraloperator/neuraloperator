"""
.. _fourier_continuation :

Fourier Continuation
========================================================
An example of usage of our Fourier continuation layer on 1d and 2d data.
"""

# %%
# Import the library
# ------------------
# We first import our `neuralop` library and required dependencies.
import torch
import matplotlib.pyplot as plt
from neuralop.layers.fourier_continuation import FCLegendre


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Creating an example of 1D curve
# --------------------
# Here we consider sin(16x) - cos(8x), which is not periodic on the interval [0,1]

length_signal = 101   # length of the original 1D signal
add_pts = 40          # number of points to add
batch_size = 3        # the Fourier continuation layer can be applied to batches of signals

x = torch.linspace(0, 1, length_signal).repeat(batch_size,1)
f = torch.sin(16 * x) - torch.cos(8 * x)




# %%
# Extending the signal
# -----------------------------------------
# We use the Fourier continuation layer to extend the signal
# We try both extending the signal on one side (right) and on both sides (left and right)

Extension = FCLegendre(n=2, d=add_pts)
f_extend_one_side = Extension(f, dim=1, one_sided=True)
f_extend_both_sides = Extension(f, dim=1, one_sided=False)


# %%
# Plot the results
# ----------------------

# Define the extended coordinates
x_extended_one_side = torch.linspace(0, 1.4, 141) 
x_extended_both_sides = torch.linspace(-0.2, 1.2, 141) 

# Add 0.5 and -0.5 to the extended functions for visualization purposes
f_extend_one_side = f_extend_one_side + 0.5
f_extend_both_sides = f_extend_both_sides - 0.5


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x[0], f[0], 'k', label='original')
plt.plot(x_extended_one_side, f_extend_one_side[0] , 'b',label='extended_one_side')
plt.plot(x_extended_both_sides, f_extend_both_sides[0] , 'g', label='extended_both_sides')
plt.plot([0, 0], [-2.5, 2.5], '-', color='gray', lw=1)  
plt.plot([1, 1], [-2.5, 2.5], '-', color='gray', lw=1)  
plt.plot([0, 1.4], [f_extend_one_side[0,0],f_extend_one_side[0,0]], '--', color='b', lw=0.5)  
plt.plot([-0.2, 1.2], [f_extend_both_sides[0,0],f_extend_both_sides[0,0]], '--', color='g', lw=0.5) 
plt.legend()
plt.tight_layout()
plt.show()
    
    
# %%
# Creating an example of a 2D function
# --------------------
# Here we consider sin(12x) - cos(14y) + 3xy, which is not periodic on [0,1]x[0,1]

length_signal = 101   # length of the signal in each dimension
add_pts = 40          # number of points to add in each dimension
batch_size = 3        # the Fourier continuation layer can be applied to batches of signals


x = torch.linspace(0, 1, length_signal).view(1, length_signal, 1).repeat(batch_size, 1, length_signal)
y = torch.linspace(0, 1, length_signal).view(1, 1, length_signal).repeat(batch_size, length_signal, 1)
f = torch.sin(12 * x)  - torch.cos(14 * y) + 3*x*y


# %%
# Extending the signal
# -----------------------------------------
# We use the Fourier continuation layer to extend the signal
# We try both extending the signal on one side (right and bottom) and on both sides (left, right, top, and bottom)

Extension = FCLegendre(n=3, d=add_pts)
f_extend_one_side = Extension(f, dim=2, one_sided=True)
f_extend_both_sides = Extension(f, dim=2, one_sided=False)



# %%
# Plot the results
# ----------------------
# We also add black lines to deliminate the original signal

fig, axs = plt.subplots(figsize=(12,4), nrows=1, ncols=3)
axs[0].imshow(f[0])
axs[0].set_title(r"Original", fontsize=17)
axs[1].imshow(f_extend_one_side[0])
axs[1].plot([length_signal, length_signal], [0, length_signal], '-', color='k', lw=3)
axs[1].plot([0, length_signal], [length_signal, length_signal], '-', color='k', lw=3)
axs[1].set_title(r"Extended one side", fontsize=17)
axs[2].imshow(f_extend_both_sides[0])
axs[2].set_title(r"Extended both sides", fontsize=17)
axs[2].plot([add_pts//2, length_signal + add_pts//2], [add_pts//2, add_pts//2], '-', color='k', lw=3)
axs[2].plot([add_pts//2, add_pts//2], [add_pts//2, length_signal + add_pts//2], '-', color='k', lw=3)
axs[2].plot([add_pts//2, length_signal + add_pts//2], [length_signal + add_pts//2, length_signal + add_pts//2], '-', color='k', lw=3)
axs[2].plot([length_signal + add_pts//2, length_signal + add_pts//2], [add_pts//2, length_signal + add_pts//2], '-', color='k', lw=3)
for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
plt.tight_layout()
plt.show()