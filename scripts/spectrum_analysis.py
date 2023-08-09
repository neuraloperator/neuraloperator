# Python script for plotting the spectrum of the any dataset
# Original Author: Zongyi Li
# Modified by: Robert Joseph George

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

font = {'size'   : 28}
matplotlib.rc('font', **font)

from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)

# Define some variables
T = 500 # number of time steps
samples = 50
s = 16 # resolution of the dataset
S = s

# additional paramaters for the dataset
Re = 5000
index = 1
T = 100
dataset_name = "Darcy Flow"

HOME_PATH = '/home/user/'

############################################################################
dataset = torch.load(HOME_PATH + 'neuraloperator/neuralop/datasets/data/darcy_test_16.pt')
print("Original dataset keys", dataset.keys()) # This is highly depending on your dataset and its structure ['x', 'y'] (In Darcy flow)
print("Original dataset shape", dataset['x'].shape) # check the shape

# It is important to note that we want the last two dimensions to represent the spatial dimensions
# So in some cases one might have to permute the dataset after squeezing the initial dimensions as well
dataset_pred = dataset['x'].squeeze() # squeeze the dataset to remove the batch dimension or other dimensions

# Shape of the dataset
shape = dataset_pred.shape

# Define the grid size - in our case its a 2d Grid repeating, for higher dimensions this will change
# Example for 3d grid
"""
batchsize, size_x, size_y, size_z = 1, shape[0], shape[1], shape[2]
gridx = torch.tensor(np.linspace(-1, 1, size_x), dtype=torch.float)
gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
gridy = torch.tensor(np.linspace(-1, 1, size_y), dtype=torch.float)
gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
gridz = torch.tensor(np.linspace(-1, 1, size_z), dtype=torch.float)
gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
grid = torch.cat((gridx, gridy, gridz), dim=-1)
"""
batchsize, size_x, size_y = 1, shape[1], shape[2]
gridx = torch.tensor(np.linspace(-1, 1, size_x), dtype=torch.float)
gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, size_y])
gridy = torch.tensor(np.linspace(-1, 1, size_y), dtype=torch.float)
gridy = gridy.reshape(1, 1, size_y).repeat([batchsize, size_x, 1])
grid = torch.cat((gridx, gridy), dim=-1)


# ##############################################################
### FFT plot
##############################################################

# Define the function to compute the spectrum
def spectrum2(u):
    """This function computes the spectrum of a 2D signal using the Fast Fourier Transform (FFT).

    Args:
        u: A 2D signal represented as a 1D tensor with shape (T * s * s), where T is the number of time steps and s is the spatial size of the signal. T can be any number of channels that we reshape into and s * s is the spatial resolution.

    Returns:
        spectrum: A 1D numpy array of shape (s,) representing the computed spectrum.
    """
    T = u.shape[0]
    u = u.reshape(T, s, s)
    # u = torch.rfft(u, 2, normalized=False, onesided=False) - depending on your choice of normalization and such
    u = torch.fft.fft2(u)

    # 2d wavenumbers following Pytorch fft convention
    k_max = s // 2
    wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1), \
                            torch.arange(start=-k_max, end=0, step=1)), 0).repeat(s, 1)
    k_x = wavenumers.transpose(0, 1)
    k_y = wavenumers
    # Sum wavenumbers
    sum_k = torch.abs(k_x) + torch.abs(k_y)
    sum_k = sum_k.numpy()
    # Remove symmetric components from wavenumbers
    index = -1.0 * np.ones((s, s))
    index[0:k_max + 1, 0:k_max + 1] = sum_k[0:k_max + 1, 0:k_max + 1]



    spectrum = np.zeros((T, s))
    for j in range(1, s + 1):
        ind = np.where(index == j)
        spectrum[:, j - 1] =  (u[:, ind[0], ind[1]].sum(axis=1)).abs() ** 2

    spectrum = spectrum.mean(axis=0)
    return spectrum


# Generate the spectrum of the dataset
# Again only the last two dimensions have to be resolution and the first dimension is the reshaped product of all the other dimensions
truth_sp = spectrum2(dataset_pred.reshape(samples * batchsize, s, s))
np.save('truth_sp.npy', truth_sp)

# Generate the spectrum plot and set all the settings
fig, ax = plt.subplots(figsize=(10,10))

linewidth = 3
ax.set_yscale('log')

length = 16 # typically till the resolution length of the dataset
buffer = 10 # just add a buffer to the plot
k = np.arange(length + buffer) * 1.0
ax.plot(truth_sp, 'k', linestyle=":", label="NS", linewidth=4)

ax.set_xlim(1,length+buffer)
ax.set_ylim(10, 10^10)
plt.legend(prop={'size': 20})
plt.title('Spectrum of {} Datset'.format(dataset_name))

plt.xlabel('wavenumber')
plt.ylabel('energy')

leg = plt.legend(loc='best')
leg.get_frame().set_alpha(0.5)
plt.savefig('darcy_flow_spectrum.png')