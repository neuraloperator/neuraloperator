"""
.. _fourier_continuation :

Fourier Continuation
========================================================
An example of usage of our Fourier continuation layer on 1d, 2d, and 3d data.
"""

# %%
# Import the library
# ------------------
# We first import our `neuralop` library and required dependencies.
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from neuralop.layers.fourier_continuation import FCLegendre,  FCGram


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Creating an example of 1D curve
# --------------------
# Here we consider sin(16x) - cos(8x), which is not periodic on the interval [0,1]

length_signal = 101   # length of the original 1D signal
add_pts = 50          # number of points to add
batch_size = 3        # the Fourier continuation layer can be applied to batches of signals

x = torch.linspace(0, 1, length_signal).repeat(batch_size,1)
f = torch.sin(16 * x) - torch.cos(8 * x)




# %%
# Extending the signal
# -----------------------------------------
# We use the FC-Legendre and FC-Gram Fourier continuation layers to extend the signal. 
# We try both extending the signal on one side (right) and on both sides (left and right)

Extension_Legendre = FCLegendre(d=2, n_additional_pts=add_pts)
f_extend_one_side_Legendre = Extension_Legendre(f, dim=1, one_sided=True)
f_extend_both_sides_Legendre = Extension_Legendre(f, dim=1, one_sided=False)

Extension_Gram = FCGram(d=3, n_additional_pts=add_pts)
f_extend_one_side_Gram = Extension_Gram(f, dim=1, one_sided=True)
f_extend_both_sides_Gram = Extension_Gram(f, dim=1, one_sided=False)


# %%
# Plot the FC-Legendre results
# ----------------------

# Define the extended coordinates
x_extended_one_side = torch.linspace(0, 1.5, 151) 
x_extended_both_sides = torch.linspace(-0.25, 1.25, 151) 

# Add 0.6 and -0.6 to the extended functions for visualization purposes
f_extend_one_side_Legendre = f_extend_one_side_Legendre + 0.6
f_extend_both_sides_Legendre = f_extend_both_sides_Legendre - 0.6

plt.figure(figsize=(14, 5))
plt.plot(x[0], f[0], 'k', label='Original Function', lw=2.2)
plt.plot(x_extended_one_side, f_extend_one_side_Legendre[0] , 'b',label='One-sided Extension', lw=2.2)
plt.plot(x_extended_both_sides, f_extend_both_sides_Legendre[0] , 'g', label='Two-sided Extension', lw=2.2)
plt.plot([0, 0], [-2.7, 1.8], '-', color='gray', lw=1.5)  
plt.plot([1, 1], [-2.7, 1.8], '-', color='gray', lw=1.5)  
plt.plot([0, 1.5], [f_extend_one_side_Legendre[0,0],f_extend_one_side_Legendre[0,0]], '--', color='b', lw=1.4)  
plt.plot([-0.25, 1.25], [f_extend_both_sides_Legendre[0,0],f_extend_both_sides_Legendre[0,0]], '--', color='g', lw=1.4) 
legend_elements = [
        Line2D([0], [0], color='none', label='FC-Legendre'),
        Line2D([0], [0], color='k', lw=2.2, label='Original Function'),
        Line2D([0], [0], color='b', lw=2.2, label='One-sided Extension'),
        Line2D([0], [0], color='g', lw=2.2, label='Two-sided Extension')
]
legend = plt.legend(handles=legend_elements, fontsize=19)
plt.xlim([-0.28, 1.56])
plt.ylim([-3.05, 2.7])
legend.get_texts()[0].set_fontweight('bold')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', which='major', labelsize=19)
ax.tick_params(axis='y', which='major', labelsize=19)
plt.xticks([-0.25,0,1,1.25, 1.5])
plt.yticks([-2,2])
plt.tight_layout()
plt.show()



# %%
# Plot the FC-Gram results
# ----------------------

# Define the extended coordinates
x_extended_one_side = torch.linspace(0, 1.5, 151) 
x_extended_both_sides = torch.linspace(-0.25, 1.25, 151) 

# Add 0.6 and -0.6 to the extended functions for visualization purposes
f_extend_one_side_Gram = f_extend_one_side_Gram + 0.6
f_extend_both_sides_Gram = f_extend_both_sides_Gram - 0.6

plt.figure(figsize=(14, 5))
plt.plot(x[0], f[0], 'k', label='Original Function', lw=2.2)
plt.plot(x_extended_one_side, f_extend_one_side_Gram[0] , 'b',label='One-sided Extension', lw=2.2)
plt.plot(x_extended_both_sides, f_extend_both_sides_Gram[0] , 'g', label='Two-sided Extension', lw=2.2)
plt.plot([0, 0], [-2.7, 1.8], '-', color='gray', lw=1.5)  
plt.plot([1, 1], [-2.7, 1.8], '-', color='gray', lw=1.5)  
plt.plot([0, 1.5], [f_extend_one_side_Gram[0,0],f_extend_one_side_Gram[0,0]], '--', color='b', lw=1.4)  
plt.plot([-0.25, 1.25], [f_extend_both_sides_Gram[0,0],f_extend_both_sides_Gram[0,0]], '--', color='g', lw=1.4) 
legend_elements = [
        Line2D([0], [0], color='none', label='FC-Gram'),
        Line2D([0], [0], color='k', lw=2.2, label='Original Function'),
        Line2D([0], [0], color='b', lw=2.2, label='One-sided Extension'),
        Line2D([0], [0], color='g', lw=2.2, label='Two-sided Extension')
]
legend = plt.legend(handles=legend_elements, fontsize=19)
plt.xlim([-0.28, 1.56])
plt.ylim([-2.95, 3.1])
legend.get_texts()[0].set_fontweight('bold')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', which='major', labelsize=19)
ax.tick_params(axis='y', which='major', labelsize=19)
plt.xticks([-0.25,0,1,1.25,1.5])
plt.yticks([-2,2])
plt.tight_layout()
plt.show()





    
# %%
# Creating an example of a 2D function
# --------------------
# Here we consider sin(12x) - cos(14y) + 3xy, which is not periodic on [0,1]x[0,1]

length_signal = 101   # length of the signal in each dimension
add_pts = 50          # number of points to add in each dimension
batch_size = 3        # the Fourier continuation layer can be applied to batches of signals


x = torch.linspace(0, 1, length_signal).view(1, length_signal, 1).repeat(batch_size, 1, length_signal)
y = torch.linspace(0, 1, length_signal).view(1, 1, length_signal).repeat(batch_size, length_signal, 1)
f = torch.sin(12 * x)  - torch.cos(14 * y) + 3*x*y


# %%
# Extending the signal
# -----------------------------------------
# We use the FC-Legendre and FC-Gram Fourier continuation layers to extend the signal.
# We try both extending the signal on one side (right and bottom) and on both sides (left, right, top, and bottom)

Extension_Legendre = FCLegendre(d=3, n_additional_pts=add_pts)
f_extend_one_side_Legendre = Extension_Legendre(f, dim=2, one_sided=True)
f_extend_both_sides_Legendre = Extension_Legendre(f, dim=2, one_sided=False)

Extension_Gram = FCGram(d=3, n_additional_pts=add_pts)
f_extend_one_side_Gram = Extension_Gram(f, dim=2, one_sided=True)
f_extend_both_sides_Gram = Extension_Gram(f, dim=2, one_sided=False)


# %%
# Plot the FC-Legendre results
# ----------------------
# We also add black lines to deliminate the original signal

fig, axs = plt.subplots(figsize=(14,6), nrows=1, ncols=3)
axs[0].imshow(f[0])
axs[0].set_title(r"Original Function", fontsize=15.5)
axs[1].imshow(f_extend_one_side_Legendre[0])
axs[1].plot([length_signal, length_signal], [0, length_signal], '-', color='k', lw=3)
axs[1].plot([0, length_signal], [length_signal, length_signal], '-', color='k', lw=3)
axs[1].set_title(r"FC-Legendre One-sided Extension", fontsize=15.5)
axs[2].set_title(r"FC-Legendre Two-sided Extension", fontsize=15.5)
axs[2].imshow(f_extend_both_sides_Legendre[0])
axs[2].plot([add_pts//2, length_signal + add_pts//2], [add_pts//2, add_pts//2], '-', color='k', lw=3)
axs[2].plot([add_pts//2, add_pts//2], [add_pts//2, length_signal + add_pts//2], '-', color='k', lw=3)
axs[2].plot([add_pts//2, length_signal + add_pts//2], [length_signal + add_pts//2, length_signal + add_pts//2], '-', color='k', lw=3)
axs[2].plot([length_signal + add_pts//2, length_signal + add_pts//2], [add_pts//2, length_signal + add_pts//2], '-', color='k', lw=3)
for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()


# %%
# Plot the FC-Gram results
# ----------------------
# We also add black lines to deliminate the original signal

fig, axs = plt.subplots(figsize=(14,6), nrows=1, ncols=3)
axs[0].imshow(f[0])
axs[0].set_title(r"Original Function", fontsize=15.5)
axs[1].imshow(f_extend_one_side_Gram[0])
axs[1].plot([length_signal, length_signal], [0, length_signal], '-', color='k', lw=3)
axs[1].plot([0, length_signal], [length_signal, length_signal], '-', color='k', lw=3)
axs[1].set_title(r"FC-Gram One-sided Extension", fontsize=15.5)
axs[2].set_title(r"FC-Gram Two-sided Extension", fontsize=15.5)
axs[2].imshow(f_extend_both_sides_Gram[0])
axs[2].plot([add_pts//2, length_signal + add_pts//2], [add_pts//2, add_pts//2], '-', color='k', lw=3)
axs[2].plot([add_pts//2, add_pts//2], [add_pts//2, length_signal + add_pts//2], '-', color='k', lw=3)
axs[2].plot([add_pts//2, length_signal + add_pts//2], [length_signal + add_pts//2, length_signal + add_pts//2], '-', color='k', lw=3)
axs[2].plot([length_signal + add_pts//2, length_signal + add_pts//2], [add_pts//2, length_signal + add_pts//2], '-', color='k', lw=3)
for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()


# %%
# Creating an example of a 3D function
# --------------------
# Here we consider f(x,y,z) = exp(-2z) + 2xz + sin(12xy) + y sin(10yz) 
# which is not periodic on [0,1]x[0,1]x[0,1]

batch_size = 2
length_signal = 101
add_pts = 50

# Create 3D grid
x = torch.linspace(0, 1, length_signal).view(1, length_signal, 1, 1).repeat(batch_size, 1, length_signal, length_signal)
y = torch.linspace(0, 1, length_signal).view(1, 1, length_signal, 1).repeat(batch_size, length_signal, 1, length_signal)
z = torch.linspace(0, 1, length_signal).view(1, 1, 1, length_signal).repeat(batch_size, length_signal, length_signal, 1)

# Create 3D function
f = torch.exp(-2*z) + 2*z*x + torch.sin(12*x*y) + y*torch.sin(10*y*z) 


# %%
# Extending the signal
# -----------------------------------------
# We use the FC-Legendre and FC-Gram Fourier continuation layers to extend the signal.
# We try both extending the signal on one side (right, bottom, back) and on both sides (left, right, top, bottom, back, front)

Extension_Legendre = FCLegendre(d=3, n_additional_pts=add_pts)
f_extend_one_side_Legendre = Extension_Legendre(f, dim=3, one_sided=True)
f_extend_both_sides_Legendre = Extension_Legendre(f, dim=3, one_sided=False)

Extension_Gram = FCGram(d=3, n_additional_pts=add_pts)
f_extend_one_side_Gram = Extension_Gram(f, dim=3, one_sided=True)
f_extend_both_sides_Gram = Extension_Gram(f, dim=3, one_sided=False)



# %%
# Plot the FC-Legendre results
# ----------------------
# We also add white lines to deliminate the original signal


f_min = f.min().item()
f_max = f.max().item()
f_ext1_min = f_extend_one_side_Legendre.min().item()
f_ext1_max = f_extend_one_side_Legendre.max().item()
f_ext2_min = f_extend_both_sides_Legendre.min().item()
f_ext2_max = f_extend_both_sides_Legendre.max().item()
global_min = min(f_min, f_ext1_min, f_ext2_min)
global_max = max(f_max, f_ext1_max, f_ext2_max)


# %%
# Figure for X slices
fig = plt.figure(figsize=(20, 15))
fig.suptitle('FC-Legendre 3D Examples: X-Slices', fontsize=24, fontweight='bold', y=0.98)
slice_indices = [length_signal//4, length_signal//2, 3*length_signal//4]
slice_names = ['First Quarter', 'Middle', 'Third Quarter']

for i, (idx, name) in enumerate(zip(slice_indices, slice_names)):
    ax = fig.add_subplot(3, 3, i+1)
    im = ax.imshow(f[0, idx, :, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    ax.set_title(f'Original: X-slice {name}', fontsize=22, fontweight='bold')
    ax.set_xlabel('Y', fontsize=20)
    ax.set_ylabel('Z', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    
    # One-sided extension - X-slice
    ax = fig.add_subplot(3, 3, i+4)
    im = ax.imshow(f_extend_one_side_Legendre[0, idx, :, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    ax.set_title(f'One-sided: X-slice {name}', fontsize=22, fontweight='bold')
    ax.set_xlabel('Y', fontsize=20)
    ax.set_ylabel('Z', fontsize=20)
    # Draw boundary lines
    ax.axhline(y=length_signal, color='white', linewidth=2, linestyle='-')
    ax.axvline(x=length_signal, color='white', linewidth=2, linestyle='-')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    
    # Two-sided extension - X-slice
    ax = fig.add_subplot(3, 3, i+7)
    ext_idx = idx + add_pts//2
    im = ax.imshow(f_extend_both_sides_Legendre[0, ext_idx, :, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    ax.set_title(f'Two-sided: X-slice {name}', fontsize=22, fontweight='bold')
    ax.set_xlabel('Y', fontsize=20)
    ax.set_ylabel('Z', fontsize=20)
    # Draw boundary lines
    ax.axhline(y=add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.axhline(y=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.axvline(x=add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.axvline(x=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)

plt.subplots_adjust(hspace=0.3, wspace=0.2, top=0.92)
plt.show()

# %%
# Figure for Y-slices
fig2 = plt.figure(figsize=(20, 15))
fig2.suptitle('FC-Legendre 3D Examples: Y-Slices', fontsize=24, fontweight='bold', y=0.98)

for i, (idx, name) in enumerate(zip(slice_indices, slice_names)):
    # Original function - Y-slice
    ax = fig2.add_subplot(3, 3, i+1)
    im = ax.imshow(f[0, :, idx, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    ax.set_title(f'Original: Y-slice {name}', fontsize=22, fontweight='bold')
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Z', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    
    # One-sided extension - Y-slice
    ax = fig2.add_subplot(3, 3, i+4)
    im = ax.imshow(f_extend_one_side_Legendre[0, :, idx, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    ax.set_title(f'One-sided: Y-slice {name}', fontsize=22, fontweight='bold')
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Z', fontsize=20)
    # Draw boundary lines
    ax.axhline(y=length_signal, color='white', linewidth=2, linestyle='-')
    ax.axvline(x=length_signal, color='white', linewidth=2, linestyle='-')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    
    # Two-sided extension - Y-slice
    ax = fig2.add_subplot(3, 3, i+7)
    ext_idx = idx + add_pts//2
    im = ax.imshow(f_extend_both_sides_Legendre[0, :, ext_idx, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    ax.set_title(f'Two-sided: Y-slice {name}', fontsize=22, fontweight='bold')
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Z', fontsize=20)
    # Draw boundary lines
    ax.axhline(y=add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.axhline(y=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.axvline(x=add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.axvline(x=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)

plt.subplots_adjust(hspace=0.3, wspace=0.2, top=0.92)
plt.show()

# %%
# Figure for Z-slices
fig3 = plt.figure(figsize=(20, 15))
fig3.suptitle('FC-Legendre 3D Examples: Z-Slices', fontsize=24, fontweight='bold', y=0.98)

for i, (idx, name) in enumerate(zip(slice_indices, slice_names)):
    # Original function - Z-slice
    ax = fig3.add_subplot(3, 3, i+1)
    im = ax.imshow(f[0, :, :, idx].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    ax.set_title(f'Original: Z-slice {name}', fontsize=22, fontweight='bold')
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    
    # One-sided extension - Z-slice
    ax = fig3.add_subplot(3, 3, i+4)
    im = ax.imshow(f_extend_one_side_Legendre[0, :, :, idx].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    ax.set_title(f'One-sided: Z-slice {name}', fontsize=22, fontweight='bold')
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    # Draw boundary lines
    ax.axhline(y=length_signal, color='white', linewidth=2, linestyle='-')
    ax.axvline(x=length_signal, color='white', linewidth=2, linestyle='-')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    
    # Two-sided extension - Z-slice
    ax = fig3.add_subplot(3, 3, i+7)
    ext_idx = idx + add_pts//2
    im = ax.imshow(f_extend_both_sides_Legendre[0, :, :, ext_idx].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    ax.set_title(f'Two-sided: Z-slice {name}', fontsize=22, fontweight='bold')
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    # Draw boundary lines
    ax.axhline(y=add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.axhline(y=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.axvline(x=add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.axvline(x=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)

plt.subplots_adjust(hspace=0.3, wspace=0.2, top=0.92)
plt.show()




# %%
# Plot the FC-Gram results
# ----------------------
# We also add white lines to deliminate the original signal

f_min = f.min().item()
f_max = f.max().item()
f_ext1_min = f_extend_one_side_Gram.min().item()
f_ext1_max = f_extend_one_side_Gram.max().item()
f_ext2_min = f_extend_both_sides_Gram.min().item()
f_ext2_max = f_extend_both_sides_Gram.max().item()

global_min = min(f_min, f_ext1_min, f_ext2_min)
global_max = max(f_max, f_ext1_max, f_ext2_max)

# %%
# Figure for X slices
fig = plt.figure(figsize=(20, 15))
fig.suptitle('FC-Gram 3D Examples: X-Slices', fontsize=24, fontweight='bold', y=0.98)
slice_indices = [length_signal//4, length_signal//2, 3*length_signal//4]
slice_names = ['First Quarter', 'Middle', 'Third Quarter']

for i, (idx, name) in enumerate(zip(slice_indices, slice_names)):
    ax = fig.add_subplot(3, 3, i+1)
    im = ax.imshow(f[0, idx, :, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    ax.set_title(f'Original: X-slice {name}', fontsize=22, fontweight='bold')
    ax.set_xlabel('Y', fontsize=20)
    ax.set_ylabel('Z', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    
    ax = fig.add_subplot(3, 3, i+4)
    im = ax.imshow(f_extend_one_side_Gram[0, idx, :, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    ax.set_title(f'One-sided: X-slice {name}', fontsize=22, fontweight='bold')
    ax.set_xlabel('Y', fontsize=20)
    ax.set_ylabel('Z', fontsize=20)
    # Draw boundary lines
    ax.axhline(y=length_signal, color='white', linewidth=2, linestyle='-')
    ax.axvline(x=length_signal, color='white', linewidth=2, linestyle='-')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    
    ax = fig.add_subplot(3, 3, i+7)
    ext_idx = idx + add_pts//2
    im = ax.imshow(f_extend_both_sides_Gram[0, ext_idx, :, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    ax.set_title(f'Two-sided: X-slice {name}', fontsize=22, fontweight='bold')
    ax.set_xlabel('Y', fontsize=20)
    ax.set_ylabel('Z', fontsize=20)
    # Draw boundary lines
    ax.axhline(y=add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.axhline(y=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.axvline(x=add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.axvline(x=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)

plt.subplots_adjust(hspace=0.3, wspace=0.2, top=0.92)
plt.show()

# %%
# Figure for Y-slices
fig2 = plt.figure(figsize=(20, 15))
fig2.suptitle('FC-Gram 3D Examples: Y-Slices', fontsize=24, fontweight='bold', y=0.98)

for i, (idx, name) in enumerate(zip(slice_indices, slice_names)):
    ax = fig2.add_subplot(3, 3, i+1)
    im = ax.imshow(f[0, :, idx, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    ax.set_title(f'Original: Y-slice {name}', fontsize=22, fontweight='bold')
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Z', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    
    ax = fig2.add_subplot(3, 3, i+4)
    im = ax.imshow(f_extend_one_side_Gram[0, :, idx, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    ax.set_title(f'One-sided: Y-slice {name}', fontsize=22, fontweight='bold')
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Z', fontsize=20)
    # Draw boundary lines
    ax.axhline(y=length_signal, color='white', linewidth=2, linestyle='-')
    ax.axvline(x=length_signal, color='white', linewidth=2, linestyle='-')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    
    ax = fig2.add_subplot(3, 3, i+7)
    ext_idx = idx + add_pts//2
    im = ax.imshow(f_extend_both_sides_Gram[0, :, ext_idx, :].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    ax.set_title(f'Two-sided: Y-slice {name}', fontsize=22, fontweight='bold')
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Z', fontsize=20)
    # Draw boundary lines
    ax.axhline(y=add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.axhline(y=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.axvline(x=add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.axvline(x=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)

plt.subplots_adjust(hspace=0.3, wspace=0.2, top=0.92)
plt.show()

# %%
# Figure for Z-slices
fig3 = plt.figure(figsize=(20, 15))
fig3.suptitle('FC-Gram 3D Examples: Z-Slices', fontsize=24, fontweight='bold', y=0.98)

for i, (idx, name) in enumerate(zip(slice_indices, slice_names)):
    ax = fig3.add_subplot(3, 3, i+1)
    im = ax.imshow(f[0, :, :, idx].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    ax.set_title(f'Original: Z-slice {name}', fontsize=22, fontweight='bold')
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    
    ax = fig3.add_subplot(3, 3, i+4)
    im = ax.imshow(f_extend_one_side_Gram[0, :, :, idx].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    ax.set_title(f'One-sided: Z-slice {name}', fontsize=22, fontweight='bold')
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    # Draw boundary lines
    ax.axhline(y=length_signal, color='white', linewidth=2, linestyle='-')
    ax.axvline(x=length_signal, color='white', linewidth=2, linestyle='-')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    
    ax = fig3.add_subplot(3, 3, i+7)
    ext_idx = idx + add_pts//2
    im = ax.imshow(f_extend_both_sides_Gram[0, :, :, ext_idx].numpy(), cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    ax.set_title(f'Two-sided: Z-slice {name}', fontsize=22, fontweight='bold')
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    # Draw boundary lines
    ax.axhline(y=add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.axhline(y=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.axvline(x=add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.axvline(x=length_signal + add_pts//2, color='white', linewidth=2, linestyle='-')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)

plt.subplots_adjust(hspace=0.3, wspace=0.2, top=0.92)
plt.show()
