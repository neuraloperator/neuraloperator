"""
Resampling layers
=================

When working with neural operators, we often need to change the resolution of our data.
For some operator architectures, like the FNO, this is handled automatically due to the 
resolution-invariant nature of the Fourier domain.

However, for other architectures, like the U-Net, we need to explicitly upsample and downsample
the data as it flows through the network. The ``neuralop.layers.resample`` function provides a 
convenient way to do this.

In this example, we'll demonstrate how to use the ``resample`` function to upsample and downsample
a sample from a Gaussian Random Field, which serves as a better visual tool than piecewise
constant data for observing the effects of interpolation.
"""
import torch
import matplotlib.pyplot as plt
from neuralop.layers.resample import resample

# %%
# First, let's generate our data. We will create a high-resolution
# Gaussian Random Field (GRF) as our ground truth. A GRF is a smooth, continuous
# signal, making it ideal for visualizing the effects of resampling.
device = 'cpu'

def generate_grf(shape, alpha=2.5, device='cpu'):
    """Generates a 2D Gaussian Random Field.
    
    Args:
        shape (tuple): The desired output shape (height, width).
        alpha (float): A parameter controlling the smoothness of the field.
                       Higher alpha leads to smoother fields.
        device (str): The device to create the tensor on.
    """
    n, m = shape
    freq_x = torch.fft.fftfreq(n, d=1/n, device=device).view(-1, 1)
    freq_y = torch.fft.fftfreq(m, d=1/m, device=device).view(1, -1)
    
    norm_sq = freq_x**2 + freq_y**2
    norm_sq[0, 0] = 1.0  # Avoid division by zero
    
    # Generate white noise in frequency domain
    noise = torch.randn(n, m, dtype=torch.cfloat, device=device)
    
    # Apply a power-law filter
    filtered_noise = noise * (norm_sq**(-alpha/2.0))
    
    # Inverse FFT to get the spatial field
    field = torch.fft.ifft2(filtered_noise).real
    
    # Normalize to [0, 1] for visualization
    field = (field - field.min()) / (field.max() - field.min())
    
    return field.unsqueeze(0).unsqueeze(0) # Add batch and channel dims

# Generate a 128x128 sample as our ground truth
high_res = 128
high_res_data = generate_grf((high_res, high_res), device=device)

# Define the low resolution we want to simulate
low_res = 32

# %%
# Now, let's use the ``resample`` function to simulate the downsampling (encoder)
# and upsampling (decoder) operations that would happen in a U-Net architecture.
# The function takes an input tensor, a `scale_factor`, and a list of `axis`
# dimensions to which the resampling is applied.

# To downsample from 128x128 to 32x32, we need a scale factor of 32/128 = 0.25
downsample_factor = low_res / high_res
downsampled_data = resample(high_res_data, downsample_factor, [2, 3])

# To upsample from 32x32 back to 128x128, we need a scale factor of 128/32 = 4
upsample_factor = high_res / low_res
upsampled_data = resample(downsampled_data, upsample_factor, [2, 3])


# %%
# Finally, let's visualize the results to see the effect of the ``resample`` function.

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Resampling a Gaussian Random Field', fontsize=16)

# Plot the original high-resolution data
im1 = axs[0].imshow(high_res_data.squeeze().cpu().numpy(), cmap='viridis')
axs[0].set_title(f'High-Resolution Ground Truth ({high_res}x{high_res})')
fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

# Plot the downsampled data
im2 = axs[1].imshow(downsampled_data.squeeze().cpu().numpy(), cmap='viridis')
axs[1].set_title(f'Downsampled by factor {downsample_factor} ({low_res}x{low_res})')
fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

# Plot the upsampled data
im3 = axs[2].imshow(upsampled_data.squeeze().cpu().numpy(), cmap='viridis')
axs[2].set_title(f'Upsampled back by factor {upsample_factor} ({high_res}x{high_res})')
fig.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)

# Hide axis ticks for a cleaner look
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %%
# As you can see, the ``resample`` function effectively changes the resolution of the data.
# Notice that the upsampled image on the right is a faithful, if slightly blurrier,
# reconstruction of the original. This is because the downsampling step is lossy;
# high-frequency details are lost and cannot be perfectly recovered. 
#
# The `neuralop.layers.resample` function uses average pooling for downsampling
# (when the scale_factor < 1), which is efficient, and bicubic interpolation for 
# upsampling, which produces a smooth result. This operation is a crucial building
# block for many neural operator architectures like the U-Net, where it forms the 
# encoder (downsampling) and decoder (upsampling) paths.