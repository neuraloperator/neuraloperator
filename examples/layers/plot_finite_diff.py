"""
.. _finite_diff :

Finite Differences
========================================================
An example of usage of Finite Differences 
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from neuralop.losses.differentiation import central_diff_1d, FiniteDiff

# %%
# 1D Finite Difference Examples
# =============================
# Here we demonstrate the central_diff_1d function

# %%
# Creating an example of 1D function
# -----------------------------------------------
# Here we consider f(x) = exp(-x) * sin(x) on [0, 2π]
L_x = 2 * torch.pi
nx = 256
x = torch.linspace(0, L_x, nx, dtype=torch.float64)

f_1d = torch.exp(-x) * torch.sin(x)

# %%
# Differentiate the 1D signal 
# ----------------------------------------------------------------
# We use the FiniteDiff class with dim=1
h = L_x / nx

# Compute derivatives
fd1d = FiniteDiff(dim=1, h=h, periodic_in_x=False)
df_dx = fd1d.dx(f_1d)
d2f_dx2 = fd1d.dx(f_1d, order=2)

# Expected analytical results for f(x) = exp(-x) * sin(x)
df_dx_expected = torch.exp(-x) * (torch.cos(x) - torch.sin(x))  # ∂f/∂x 
d2f_dx2_expected = torch.exp(-x) * (-2*torch.cos(x))  # ∂²f/∂x²

# %%
# Plot the 1D results
# ---------------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 18))
fig.suptitle('1D Finite Differences: f(x) = exp(-x) * sin(x)')

# Original function
axes[0].plot(x.cpu().numpy(), f_1d.cpu().numpy(), 'b-', linewidth=1.5)
axes[0].set_title('Original: exp(-x) * sin(x)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')

# First derivative
axes[1].plot(x.cpu().numpy(), df_dx.cpu().numpy(), 'r-', linewidth=1.5, label='Computed')
axes[1].plot(x.cpu().numpy(), df_dx_expected.cpu().numpy(), 'r--', linewidth=2, label='Expected: exp(-x) * (cos(x) - sin(x))')
axes[1].set_title('∂f/∂x')
axes[1].set_xlabel('x')
axes[1].set_ylabel('∂f/∂x')
axes[1].legend()

# Second derivative
axes[2].plot(x.cpu().numpy(), d2f_dx2.cpu().numpy(), 'g-', linewidth=1.5, label='Computed')
axes[2].plot(x.cpu().numpy(), d2f_dx2_expected.cpu().numpy(), 'g--', linewidth=2, label='Expected: exp(-x) * (-2cos(x))')
axes[2].set_title('∂²f/∂x²')
axes[2].set_xlabel('x')
axes[2].set_ylabel('∂²f/∂x²')
axes[2].legend()

plt.tight_layout()
plt.show()

# %%
# 2D Finite Difference Examples
# =============================
# Here we demonstrate the FiniteDiff class for 2D functions

# %% fix 
# Creating an example of 2D function
# -----------------------------------------------
# Here we consider f(x,y) = exp(-x) * sin(y), which is non-periodic on [0, 2π] × [0, 2π]
L_x, L_y = 2 * torch.pi, 2 * torch.pi
nx, ny = 256, 256
x = torch.linspace(0, L_x, nx, dtype=torch.float64)
y = torch.linspace(0, L_y, ny, dtype=torch.float64)
X, Y = torch.meshgrid(x, y, indexing='ij')

# Test function: f(x,y) = exp(-x) * sin(y)
f_2d = torch.exp(-X) * torch.sin(Y)

# %%
# Differentiate the 2D signal 
# ----------------------------------------------------------------
# We use the FiniteDiff class with dim=2 to compute derivatives
fd2d = FiniteDiff(dim=2, h=(L_x/nx, L_y/ny), periodic_in_x=False, periodic_in_y=False)

# Compute derivatives
df_dx = fd2d.dx(f_2d)
df_dy = fd2d.dy(f_2d)
d2f_dx2 = fd2d.dx(f_2d, order=2)
d2f_dy2 = fd2d.dy(f_2d, order=2)
laplacian = fd2d.laplacian(f_2d)

# Expected analytical results for f(x,y) = exp(-x) * sin(y)
df_dx_expected = -torch.exp(-X) * torch.sin(Y)  # ∂f/∂x 
df_dy_expected = torch.exp(-X) * torch.cos(Y)   # ∂f/∂y
d2f_dx2_expected = torch.exp(-X) * torch.sin(Y)  # ∂²f/∂x²
d2f_dy2_expected = -torch.exp(-X) * torch.sin(Y)  # ∂²f/∂y²
laplacian_expected = torch.zeros_like(X)  # ∇²f

# %%
# Plot the 2D results
# ---------------------------------
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('2D Finite Differences: f(x,y) = exp(-x) * sin(y)')

# Compute consistent colorbar limits for each derivative pair
df_dx_min = min(df_dx.min().item(), df_dx_expected.min().item())
df_dx_max = max(df_dx.max().item(), df_dx_expected.max().item())
df_dy_min = min(df_dy.min().item(), df_dy_expected.min().item())
df_dy_max = max(df_dy.max().item(), df_dy_expected.max().item())

# Compute consistent colorbar limits for second derivatives
d2f_dx2_min = min(d2f_dx2.min().item(), d2f_dx2_expected.min().item())
d2f_dx2_max = max(d2f_dx2.max().item(), d2f_dx2_expected.max().item())
d2f_dy2_min = min(d2f_dy2.min().item(), d2f_dy2_expected.min().item())
d2f_dy2_max = max(d2f_dy2.max().item(), d2f_dy2_expected.max().item())

# Compute consistent colorbar limits for laplacian
laplacian_min = min(laplacian.min().item(), laplacian_expected.min().item())
laplacian_max = max(laplacian.max().item(), laplacian_expected.max().item())

# Original function
im0 = axes[0, 0].imshow(f_2d.cpu().numpy())
axes[0, 0].set_title('Original: exp(-x) * sin(y)')
plt.colorbar(im0, ax=axes[0, 0], shrink=0.62)

# ∂f/∂x computed
im1 = axes[0, 1].imshow(df_dx.cpu().numpy(), vmin=df_dx_min, vmax=df_dx_max)
axes[0, 1].set_title('∂f/∂x (computed)')
plt.colorbar(im1, ax=axes[0, 1], shrink=0.62)

# ∂f/∂x expected
im2 = axes[0, 2].imshow(df_dx_expected.cpu().numpy(), vmin=df_dx_min, vmax=df_dx_max)
axes[0, 2].set_title('∂f/∂x (expected: -exp(-x) * sin(y))')
plt.colorbar(im2, ax=axes[0, 2], shrink=0.62)

# ∂f/∂y computed
im3 = axes[0, 3].imshow(df_dy.cpu().numpy(), vmin=df_dy_min, vmax=df_dy_max)
axes[0, 3].set_title('∂f/∂y (computed)')
plt.colorbar(im3, ax=axes[0, 3], shrink=0.62)

# ∂f/∂y expected
im4 = axes[1, 0].imshow(df_dy_expected.cpu().numpy(), vmin=df_dy_min, vmax=df_dy_max)
axes[1, 0].set_title('∂f/∂y (expected: exp(-x) * cos(y))')
plt.colorbar(im4, ax=axes[1, 0], shrink=0.62)

# Laplacian computed
im5 = axes[1, 1].imshow(laplacian.cpu().numpy(), vmin=laplacian_min, vmax=laplacian_max)
axes[1, 1].set_title('∇²f (computed)')
plt.colorbar(im5, ax=axes[1, 1], shrink=0.62)

# Laplacian expected
im6 = axes[1, 2].imshow(laplacian_expected.cpu().numpy(), vmin=laplacian_min, vmax=laplacian_max)
axes[1, 2].set_title('∇²f (expected: 0)')
plt.colorbar(im6, ax=axes[1, 2], shrink=0.62)

# Error in laplacian
error = torch.abs(laplacian - laplacian_expected)
im7 = axes[1, 3].imshow(error.cpu().numpy())
axes[1, 3].set_title('Error in ∇²f')
plt.colorbar(im7, ax=axes[1, 3], shrink=0.62)

plt.tight_layout()
plt.show()

# %%
# Test gradient computation
# -------------------------
# Compute gradient of the scalar field
gradient = fd2d.gradient(f_2d)  # Returns [df_dx, df_dy]

# Plot gradient components
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Gradient Components: ∇f = [∂f/∂x, ∂f/∂y]')

# ∂f/∂x from gradient
im0 = axes[0, 0].imshow(gradient[0].cpu().numpy(), vmin=df_dx_min, vmax=df_dx_max)
axes[0, 0].set_title('∂f/∂x from gradient')
plt.colorbar(im0, ax=axes[0, 0], shrink=0.62)

# ∂f/∂y from gradient
im1 = axes[0, 1].imshow(gradient[1].cpu().numpy(), vmin=df_dy_min, vmax=df_dy_max)
axes[0, 1].set_title('∂f/∂y from gradient')
plt.colorbar(im1, ax=axes[0, 1], shrink=0.62)

# Compare with direct computation
im2 = axes[1, 0].imshow((gradient[0] - df_dx).cpu().numpy())
axes[1, 0].set_title('Difference: gradient[0] - df_dx')
plt.colorbar(im2, ax=axes[1, 0], shrink=0.62)

im3 = axes[1, 1].imshow((gradient[1] - df_dy).cpu().numpy())
axes[1, 1].set_title('Difference: gradient[1] - df_dy')
plt.colorbar(im3, ax=axes[1, 1], shrink=0.62)

plt.tight_layout()
plt.show()

# %%
# Test vector field operations
# -----------------------------
# Create a vector field: u = [exp(-x), sin(y)]
u1 = torch.exp(-X)
u2 = torch.sin(Y)
u_vector = torch.stack([u1, u2], dim=0)

# Compute divergence and curl
divergence = fd2d.divergence(u_vector)
curl = fd2d.curl(u_vector)

# Expected analytical results
# ∇·u = ∂u₁/∂x + ∂u₂/∂y = -exp(-x) + cos(y)
divergence_expected = -torch.exp(-X) + torch.cos(Y)
# ∇×u = ∂u₂/∂x - ∂u₁/∂y = 0 - 0 = 0 (since u₁ doesn't depend on y, u₂ doesn't depend on x)
curl_expected = torch.zeros_like(X)

# %%
# Plot vector field operations
# -----------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Vector Field Operations: u = [exp(-x), sin(y)]')

# Compute consistent colorbar limits for vector field components
u1_min = min(u1.min().item(), u1.max().item())
u1_max = max(u1.min().item(), u1.max().item())
u2_min = min(u2.min().item(), u2.max().item())
u2_max = max(u2.min().item(), u2.max().item())

# Compute consistent colorbar limits for divergence
div_min = min(divergence.min().item(), divergence_expected.min().item())
div_max = max(divergence.max().item(), divergence_expected.max().item())

# Compute consistent colorbar limits for curl
curl_min = min(curl.min().item(), curl_expected.min().item())
curl_max = max(curl.max().item(), curl_expected.max().item())

# Vector field components
im0 = axes[0, 0].imshow(u1.cpu().numpy(), vmin=u1_min, vmax=u1_max)
axes[0, 0].set_title('u₁ = exp(-x)')
plt.colorbar(im0, ax=axes[0, 0], shrink=0.62)

im1 = axes[0, 1].imshow(u2.cpu().numpy(), vmin=u2_min, vmax=u2_max)
axes[0, 1].set_title('u₂ = sin(y)')
plt.colorbar(im1, ax=axes[0, 1], shrink=0.62)

# Divergence
im2 = axes[0, 2].imshow(divergence.cpu().numpy(), vmin=div_min, vmax=div_max)
axes[0, 2].set_title('∇·u (computed)')
plt.colorbar(im2, ax=axes[0, 2], shrink=0.62)

# Divergence expected
im3 = axes[1, 0].imshow(divergence_expected.cpu().numpy(), vmin=div_min, vmax=div_max)
axes[1, 0].set_title('∇·u (expected: -exp(-x) + cos(y))')
plt.colorbar(im3, ax=axes[1, 0], shrink=0.62)

# Curl
im4 = axes[1, 1].imshow(curl.cpu().numpy(), vmin=curl_min, vmax=curl_max)
axes[1, 1].set_title('∇×u (computed)')
plt.colorbar(im4, ax=axes[1, 1], shrink=0.62)

# Curl expected
im5 = axes[1, 2].imshow(curl_expected.cpu().numpy(), vmin=curl_min, vmax=curl_max)
axes[1, 2].set_title('∇×u (expected: 0)')
plt.colorbar(im5, ax=axes[1, 2], shrink=0.62)

plt.tight_layout()
plt.show()

# %%
# Additional verification plots
# -----------------------------
# Show second derivatives with consistent colorbars
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Second Derivatives: ∂²f/∂x² and ∂²f/∂y²')

# ∂²f/∂x² computed
im0 = axes[0, 0].imshow(d2f_dx2.cpu().numpy(), vmin=d2f_dx2_min, vmax=d2f_dx2_max)
axes[0, 0].set_title('∂²f/∂x² (computed)')
plt.colorbar(im0, ax=axes[0, 0], shrink=0.62)

# ∂²f/∂x² expected
im1 = axes[0, 1].imshow(d2f_dx2_expected.cpu().numpy(), vmin=d2f_dx2_min, vmax=d2f_dx2_max)
axes[0, 1].set_title('∂²f/∂x² (expected: exp(-x) * sin(y))')
plt.colorbar(im1, ax=axes[0, 1], shrink=0.62)

# ∂²f/∂y² computed
im2 = axes[1, 0].imshow(d2f_dy2.cpu().numpy(), vmin=d2f_dy2_min, vmax=d2f_dy2_max)
axes[1, 0].set_title('∂²f/∂y² (computed)')
plt.colorbar(im2, ax=axes[1, 0], shrink=0.62)

# ∂²f/∂y² expected
im3 = axes[1, 1].imshow(d2f_dy2_expected.cpu().numpy(), vmin=d2f_dy2_min, vmax=d2f_dy2_max)
axes[1, 1].set_title('∂²f/∂y² (expected: -exp(-x) * sin(y))')
plt.colorbar(im3, ax=axes[1, 1], shrink=0.62)

plt.tight_layout()
plt.show()

# %%
# 3D Finite Difference Examples
# =============================
# Here we demonstrate the FiniteDiff class for 3D functions

# %%
# Creating an example of 3D function
# -----------------------------------------------
# Here we consider f(x,y,z) = exp(-x) * sin(y) * cos(z), which is on [0, 2π]³
L_x, L_y, L_z = 2 * torch.pi, 2 * torch.pi, 2 * torch.pi
nx, ny, nz = 80, 84, 76
x = torch.linspace(0, L_x, nx, dtype=torch.float64)
y = torch.linspace(0, L_y, ny, dtype=torch.float64)
z = torch.linspace(0, L_z, nz, dtype=torch.float64)
X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

# Test function: f(x,y,z) = exp(-x) * sin(y) * cos(z)
f_3d = torch.exp(-X) * torch.sin(Y) * torch.cos(Z)

# %%
# Differentiate the 3D signal 
# ----------------------------------------------------------------
# We use the FiniteDiff class with dim=3 to compute derivatives
fd3d = FiniteDiff(dim=3, h=(L_x/nx, L_y/ny, L_z/nz), periodic_in_x=False, periodic_in_y=True, periodic_in_z=True)

# Compute derivatives
df_dx = fd3d.dx(f_3d)
df_dy = fd3d.dy(f_3d)
df_dz = fd3d.dz(f_3d)
d2f_dx2 = fd3d.dx(f_3d, order=2)
d2f_dy2 = fd3d.dy(f_3d, order=2)
d2f_dz2 = fd3d.dz(f_3d, order=2)
laplacian_3d = fd3d.laplacian(f_3d)

# Expected analytical results for f(x,y,z) = exp(-x) * sin(y) * cos(z)
df_dx_expected = -torch.exp(-X) * torch.sin(Y) * torch.cos(Z)  # ∂f/∂x 
df_dy_expected = torch.exp(-X) * torch.cos(Y) * torch.cos(Z)   # ∂f/∂y 
df_dz_expected = -torch.exp(-X) * torch.sin(Y) * torch.sin(Z)  # ∂f/∂z 

d2f_dx2_expected = torch.exp(-X) * torch.sin(Y) * torch.cos(Z)   # ∂²f/∂x² 
d2f_dy2_expected = -torch.exp(-X) * torch.sin(Y) * torch.cos(Z)  # ∂²f/∂y² 
d2f_dz2_expected = -torch.exp(-X) * torch.sin(Y) * torch.cos(Z)  # ∂²f/∂z² 

# Laplacian: ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z² 
laplacian_3d_expected = -torch.exp(-X) * torch.sin(Y) * torch.cos(Z)

# %%
# Plot 3D results at a specific z-slice
# -------------------------------------
z_slice_idx = nz // 2  # Middle z-slice
z_slice_val = z[z_slice_idx].item()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'3D Finite Differences: f(x,y,z) = exp(-x) * sin(y) * cos(z) at z = {z_slice_val:.2f}')

# Compute consistent colorbar limits for each derivative pair at the z-slice
df_dx_3d_slice = df_dx[:, :, z_slice_idx]
df_dx_expected_3d_slice = df_dx_expected[:, :, z_slice_idx]
df_dy_3d_slice = df_dy[:, :, z_slice_idx]
df_dy_expected_3d_slice = df_dy_expected[:, :, z_slice_idx]
df_dz_3d_slice = df_dz[:, :, z_slice_idx]
df_dz_expected_3d_slice = df_dz_expected[:, :, z_slice_idx]

df_dx_3d_min = min(df_dx_3d_slice.min().item(), df_dx_expected_3d_slice.min().item())
df_dx_3d_max = max(df_dx_3d_slice.max().item(), df_dx_expected_3d_slice.max().item())
df_dy_3d_min = min(df_dy_3d_slice.min().item(), df_dy_expected_3d_slice.min().item())
df_dy_3d_max = max(df_dy_3d_slice.max().item(), df_dy_expected_3d_slice.max().item())
df_dz_3d_min = min(df_dz_3d_slice.min().item(), df_dz_expected_3d_slice.min().item())
df_dz_3d_max = max(df_dz_3d_slice.max().item(), df_dz_expected_3d_slice.max().item())

# Original function at z-slice
im0 = axes[0, 0].imshow(f_3d[:, :, z_slice_idx].cpu().numpy())
axes[0, 0].set_title(f'Original: exp(-x) * sin(y) * cos(z) at z = {z_slice_val:.2f}')
plt.colorbar(im0, ax=axes[0, 0], shrink=0.62)

# ∂f/∂x computed
im1 = axes[0, 1].imshow(df_dx_3d_slice.cpu().numpy(), vmin=df_dx_3d_min, vmax=df_dx_3d_max)
axes[0, 1].set_title('∂f/∂x (computed)')
plt.colorbar(im1, ax=axes[0, 1], shrink=0.62)

# ∂f/∂x expected
im2 = axes[0, 2].imshow(df_dx_expected_3d_slice.cpu().numpy(), vmin=df_dx_3d_min, vmax=df_dx_3d_max)
axes[0, 2].set_title('∂f/∂x (expected: -exp(-x) * sin(y) * cos(z))')
plt.colorbar(im2, ax=axes[0, 2], shrink=0.62)

# ∂f/∂y computed
im3 = axes[1, 0].imshow(df_dy_3d_slice.cpu().numpy(), vmin=df_dy_3d_min, vmax=df_dy_3d_max)
axes[1, 0].set_title('∂f/∂y (computed)')
plt.colorbar(im3, ax=axes[1, 0], shrink=0.62)

# ∂f/∂y expected
im4 = axes[1, 1].imshow(df_dy_expected_3d_slice.cpu().numpy(), vmin=df_dy_3d_min, vmax=df_dy_3d_max)
axes[1, 1].set_title('∂f/∂y (expected: exp(-x) * cos(y) * cos(z))')
plt.colorbar(im4, ax=axes[1, 1], shrink=0.62)

# ∂f/∂z expected
im5 = axes[1, 2].imshow(df_dz_expected_3d_slice.cpu().numpy(), vmin=df_dz_3d_min, vmax=df_dz_3d_max)
axes[1, 2].set_title('∂f/∂z (expected: -exp(-x) * sin(y) * sin(z))')
plt.colorbar(im5, ax=axes[1, 2], shrink=0.62)

plt.tight_layout()
plt.show()

# %%
# Test 3D gradient computation
# -----------------------------
# Compute gradient of the 3D scalar field
gradient_3d = fd3d.gradient(f_3d)  # Returns [df_dx, df_dy, df_dz]

# Plot gradient components at z-slice
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'3D Gradient Components: ∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z] at z = {z_slice_val:.2f}')

# ∂f/∂x from gradient
im0 = axes[0, 0].imshow(gradient_3d[0][:, :, z_slice_idx].cpu().numpy(), vmin=df_dx_3d_min, vmax=df_dx_3d_max)
axes[0, 0].set_title('∂f/∂x from gradient')
plt.colorbar(im0, ax=axes[0, 0], shrink=0.62)

# ∂f/∂y from gradient
im1 = axes[0, 1].imshow(gradient_3d[1][:, :, z_slice_idx].cpu().numpy(), vmin=df_dy_3d_min, vmax=df_dy_3d_max)
axes[0, 1].set_title('∂f/∂y from gradient')
plt.colorbar(im1, ax=axes[0, 1], shrink=0.62)

# ∂f/∂z from gradient
im2 = axes[0, 2].imshow(gradient_3d[2][:, :, z_slice_idx].cpu().numpy(), vmin=df_dz_3d_min, vmax=df_dz_3d_max)
axes[0, 2].set_title('∂f/∂z from gradient')
plt.colorbar(im2, ax=axes[0, 2], shrink=0.62)

# Reference gradient components (expected values)
im3 = axes[1, 0].imshow(df_dx_expected_3d_slice.cpu().numpy(), vmin=df_dx_3d_min, vmax=df_dx_3d_max)
axes[1, 0].set_title('∂f/∂x (expected)')
plt.colorbar(im3, ax=axes[1, 0], shrink=0.62)

im4 = axes[1, 1].imshow(df_dy_expected_3d_slice.cpu().numpy(), vmin=df_dy_3d_min, vmax=df_dy_3d_max)
axes[1, 1].set_title('∂f/∂y (expected)')
plt.colorbar(im4, ax=axes[1, 1], shrink=0.62)

im5 = axes[1, 2].imshow(df_dz_expected_3d_slice.cpu().numpy(), vmin=df_dz_3d_min, vmax=df_dz_3d_max)
axes[1, 2].set_title('∂f/∂z (expected)')
plt.colorbar(im5, ax=axes[1, 2], shrink=0.62)

plt.tight_layout()
plt.show()

# %%
# Test 3D vector field operations
# --------------------------------
# Create a 3D vector field: u = [exp(-x), sin(y), cos(z)]
u1_3d = torch.exp(-X)
u2_3d = torch.sin(Y)
u3_3d = torch.cos(Z)
u_vector_3d = torch.stack([u1_3d, u2_3d, u3_3d], dim=0)

# Compute divergence
divergence_3d = fd3d.divergence(u_vector_3d)

# Expected analytical results
# ∇·u = ∂u₁/∂x + ∂u₂/∂y + ∂u₃/∂z 
divergence_3d_expected = -torch.exp(-X) + torch.cos(Y) - torch.sin(Z)

# %%
# Plot 3D vector field operations at z-slice
# -------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'3D Vector Field Operations: u = [exp(-x), sin(y), cos(z)] at z = {z_slice_val:.2f}')

# Compute consistent colorbar limits for vector field components at z-slice
u1_3d_slice = u1_3d[:, :, z_slice_idx]
u2_3d_slice = u2_3d[:, :, z_slice_idx]
u3_3d_slice = u3_3d[:, :, z_slice_idx]

u1_3d_min = min(u1_3d_slice.min().item(), u1_3d_slice.max().item())
u1_3d_max = max(u1_3d_slice.min().item(), u1_3d_slice.max().item())
u2_3d_min = min(u2_3d_slice.min().item(), u2_3d_slice.max().item())
u2_3d_max = max(u2_3d_slice.min().item(), u2_3d_slice.max().item())
u3_3d_min = min(u3_3d_slice.min().item(), u3_3d_slice.max().item())
u3_3d_max = max(u3_3d_slice.min().item(), u3_3d_slice.max().item())

# Compute consistent colorbar limits for divergence at z-slice
div_3d_slice = divergence_3d[:, :, z_slice_idx]
div_3d_expected_slice = divergence_3d_expected[:, :, z_slice_idx]
div_3d_min = min(div_3d_slice.min().item(), div_3d_expected_slice.min().item())
div_3d_max = max(div_3d_slice.max().item(), div_3d_expected_slice.max().item())

# Vector field components
im0 = axes[0, 0].imshow(u1_3d_slice.cpu().numpy(), vmin=u1_3d_min, vmax=u1_3d_max)
axes[0, 0].set_title('u₁ = exp(-x)')
plt.colorbar(im0, ax=axes[0, 0], shrink=0.62)

im1 = axes[0, 1].imshow(u2_3d_slice.cpu().numpy(), vmin=u2_3d_min, vmax=u2_3d_max)
axes[0, 1].set_title('u₂ = sin(y)')
plt.colorbar(im1, ax=axes[0, 1], shrink=0.62)

im2 = axes[0, 2].imshow(u3_3d_slice.cpu().numpy(), vmin=u3_3d_min, vmax=u3_3d_max)
axes[0, 2].set_title('u₃ = cos(z)')
plt.colorbar(im2, ax=axes[0, 2], shrink=0.62)

# Divergence
im3 = axes[1, 0].imshow(div_3d_slice.cpu().numpy(), vmin=div_3d_min, vmax=div_3d_max)
axes[1, 0].set_title('∇·u (computed)')
plt.colorbar(im3, ax=axes[1, 0], shrink=0.62)

# Divergence expected
im4 = axes[1, 1].imshow(div_3d_expected_slice.cpu().numpy(), vmin=div_3d_min, vmax=div_3d_max)
axes[1, 1].set_title('∇·u (expected: -exp(-x) + cos(y) - sin(z))')
plt.colorbar(im4, ax=axes[1, 1], shrink=0.62)

# Empty plot for symmetry
axes[1, 2].set_visible(False)

plt.tight_layout()
plt.show()
