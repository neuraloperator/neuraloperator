"""
Training a neural operator on Darcy-Flow - Author Robert Joseph
========================================
In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package on Incremental FNO
"""

# %%
# 

import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop.training import OutputEncoderCallback, SimpleWandBLoggerCallback, IncrementalCallback
from neuralop.datasets import data_transforms
from neuralop import LpLoss, H1Loss
device = 'cpu'


# %%
# Loading the Darcy flow dataset
train_loader, test_loaders, output_encoder = load_darcy_flow_small(
        n_train=100, batch_size=16, 
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
)

# %%
# Choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# Set up the incremental FNO model
# We start with 2 modes in each dimension
# We choose to update the modes by the incremental gradient explained algorithm
starting_modes = (2, 2)
incremental = True

# %%
# set up model
model = FNO(max_n_modes=(32, 32), n_modes=starting_modes, hidden_channels=32, projection_channels=64)
model = model.to(device)
n_params = count_model_params(model)

# %%
# Set up the optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
data_transform = data_transforms.IncrementalDataProcessor(in_normalizer=None, out_normalizer=None, positional_encoding=None, dataset_sublist=[1], dataset_resolution=16, dataset_indices=[0])

# %%
# Set up the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}
print('\n### N PARAMS ###\n', n_params)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print('\n### INCREMENTAL GRADIENT EXPLAINED ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()

# %%
# Set up the trainer
# other options include setting incremental_loss_gap = True
# If one wants to use incremental resolution set it to True
# In this example we only update the modes and not the resolution
# When using the incremental resolution one should keep in mind that the numnber of modes initially set should be strictly less than the resolution
# Again these are the various paramaters for the various incremental settings
#incremental : bool, default is False
#    if True, use the base incremental algorithm which is based on gradient variance
#    uses the incremental_grad_eps parameter - set the threshold for gradient variance
#    uses the incremental_buffer paramater - sets the number of buffer modes to calculate the gradient variance
#    uses the incremental_max_iter parameter - sets the initial number of iterations
#    uses the incremental_grad_max_iter parameter - sets the maximum number of iterations to accumulate the gradients
#incremental_loss_gap : bool, default is False
#    if True, use the incremental algorithm based on loss gap
#    uses the incremental_loss_eps parameter
#incremental_resolution : bool, default is False
#    if True, increase the resolution of the input incrementally
#    uses the incremental_res_gap parameter
#    uses the dataset_sublist parameter - a list of resolutions to use
#    uses the dataset_indices parameter - a list of indices of the dataset to slice to regularize the input resolution
#    uses the dataset_resolution parameter - the resolution of the input

callbacks = [IncrementalCallback(incremental_grad_eps=0.001, incremental_buffer=5, incremental_max_iter=1, incremental_grad_max_iter=10, incremental_loss_eps=0.001, incremental_res_gap=0.001)]
trainer = Trainer(model, data_processor= data_transform, callbacks=callbacks, n_epochs=10, device=device, mg_patching_levels=0, wandb_log=False, log_test_interval=3,      
                  use_distributed=False, verbose=True, incremental = incremental, incremental_eps=0.001, incremental_buffer=5, incremental_max_iter=1, incremental_grad_max_iter=10)

# %% 
# Train the model
trainer.train(train_loader, test_loaders, output_encoder, model, optimizer, scheduler, regularizer=False, training_loss=train_loss, eval_losses=eval_losses)

# %%
# Check that the number of modes has dynamically increased (Atleast for these settings on this dataset it should increase)
assert model.fno_blocks.convs.n_modes > starting_modes

# %%
# Plot the prediction, and compare with the ground-truth 
# Note that we trained on a very small resolution for
# a very small number of epochs
# In practice, we would train at larger resolution, on many more samples.
# 
# However, for practicity, we created a minimal example that
# i) fits in just a few Mb of memory
# ii) can be trained quickly on CPU
#
# In practice we would train a Neural Operator on one or multiple GPUs

"""test_samples = test_loaders[32].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0))

    ax = fig.add_subplot(3, 3, index*3 + 1)
    ax.imshow(x[0], cmap='gray')
    if index == 0: 
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 2)
    ax.imshow(y.squeeze())
    if index == 0: 
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 3)
    ax.imshow(out.squeeze().detach().numpy())
    if index == 0: 
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.show()"""