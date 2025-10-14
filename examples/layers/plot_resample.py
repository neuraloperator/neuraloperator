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
a sample from the Darcy Flow dataset, a common benchmark in neural operator literature.
"""

import matplotlib.pyplot as plt
from neuralop.layers.resample import resample
from neuralop.data.datasets import load_darcy_flow_small

# %%
# First, let's load our data. We will use a high-resolution sample
# from the Darcy-Flow dataset as our ground truth.
device = 'cpu'

# Load the Darcy-Flow dataset, setting num_workers=0 to avoid multiprocessing issues
_, test_loaders, _ = load_darcy_flow_small(
        n_train=1, batch_size=1, 
        test_resolutions=[32], n_tests=[1],
        test_batch_sizes=[1]
)

data = next(iter(test_loaders[32]))
high_res_data = data['x'].to(device)

# Define the low resolution we want to simulate
low_res = 16
high_res = high_res_data.shape[-1]


# %%
# Now, let's use the ``resample`` function to simulate the downsampling (encoder)
# and upsampling (decoder) operations that would happen in a U-Net architecture.
# The function takes an input tensor, a `scale_factor`, and a list of `axis`
# dimensions to which the resampling is applied.

# To downsample from 32x32 to 16x16, we need a scale factor of 16/32 = 0.5
downsample_factor = low_res / high_res
downsampled_data = resample(high_res_data, downsample_factor, [2, 3])

# To upsample from 16x16 back to 32x32, we need a scale factor of 32/16 = 2
upsample_factor = high_res / low_res
upsampled_data = resample(downsampled_data, upsample_factor, [2, 3])


# %%
# Finally, let's visualize the results to see the effect of the ``resample`` function.

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Resampling a Darcy Flow Sample', fontsize=16)

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
# high-frequency details are lost and cannot be perfectly recovered by interpolation.
#
# This operation is a crucial building block for many neural operator architectures
# like the U-Net, where it forms the encoder (downsampling) and decoder (upsampling) paths.
