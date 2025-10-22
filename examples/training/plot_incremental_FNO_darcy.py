"""
Training an FNO with incremental meta-learning
===============================================
A demo of the Incremental FNO meta-learning algorithm on our small Darcy-Flow dataset.

This tutorial demonstrates incremental meta-learning for neural operators, which allows
the model to gradually increase its complexity during training. This approach can lead to:

- Better convergence properties
- More stable training dynamics
- Improved generalization
- Reduced computational requirements during early training

The incremental approach starts with a small number of Fourier modes and gradually
increases the model capacity as training progresses.
"""

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Import dependencies
# -------------------
# We import the necessary modules for incremental FNO training

import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import FNO
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop.training import AdamW
from neuralop.training.incremental import IncrementalFNOTrainer
from neuralop.data.transforms.data_processors import IncrementalDataProcessor
from neuralop import LpLoss, H1Loss

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Loading the Darcy-Flow dataset
# ------------------------------
# We load the Darcy-Flow dataset with multiple resolutions for incremental training.

train_loader, test_loaders, output_encoder = load_darcy_flow_small(
    n_train=100,
    batch_size=16,
    test_resolutions=[16, 32],
    n_tests=[100, 50],
    test_batch_sizes=[32, 32],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Configuring incremental training
# --------------------------------
# We set up the incremental FNO model with a small starting number of modes.
# The model will gradually increase its capacity during training.
# We choose to update the modes using the incremental gradient explained algorithm
incremental = True
if incremental:
    starting_modes = (2, 2)  # Start with very few modes
else:
    starting_modes = (8, 8)  # Standard number of modes

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Creating the incremental FNO model
# ----------------------------------
# We create an FNO model with a maximum number of modes that can be reached
# during incremental training. The model starts with fewer modes and grows.
model = FNO(
    max_n_modes=(8, 8),  # Maximum modes the model can reach
    n_modes=starting_modes,  # Starting number of modes
    hidden_channels=32,
    in_channels=1,
    out_channels=1,
)
model = model.to(device)
n_params = count_model_params(model)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Setting up the optimizer and scheduler
# -------------------------------------
# We use AdamW optimizer with weight decay for regularization
optimizer = AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Configuring incremental data processing
# ---------------------------------------
# If one wants to use Incremental Resolution, one should use the IncrementalDataProcessor.
# When passed to the trainer, the trainer will automatically update the resolution.
#
# Key parameters for incremental resolution:
#
# - incremental_resolution: bool, default is False. If True, increase the resolution of the input incrementally
# - incremental_res_gap: parameter for resolution updates
# - subsampling_rates: a list of resolutions to use
# - dataset_indices: a list of indices of the dataset to slice to regularize the input resolution
# - dataset_resolution: the resolution of the input
# - epoch_gap: the number of epochs to wait before increasing the resolution
# - verbose: if True, print the resolution and the number of modes

data_transform = IncrementalDataProcessor(
    in_normalizer=None,
    out_normalizer=None,
    device=device,
    subsampling_rates=[2, 1],  # Resolution scaling factors
    dataset_resolution=16,  # Base resolution
    dataset_indices=[2, 3],  # Dataset indices for regularization
    epoch_gap=10,  # Epochs between resolution updates
    verbose=True,  # Print progress information
)

data_transform = data_transform.to(device)
# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Setting up loss functions
# -------------------------
# We use H1 loss for training and L2 loss for evaluation
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Displaying training configuration
# ---------------------------------
# We display the model parameters, optimizer, scheduler, and loss functions
# to verify our incremental training setup
print("\n### N PARAMS ###\n", n_params)
print("\n### OPTIMIZER ###\n", optimizer)
print("\n### SCHEDULER ###\n", scheduler)
print("\n### LOSSES ###")
print("\n### INCREMENTAL RESOLUTION + GRADIENT EXPLAINED ###")
print(f"\n * Train: {train_loss}")
print(f"\n * Test: {eval_losses}")
sys.stdout.flush()

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Configuring the IncrementalFNOTrainer
# --------------------------------------
# We set up the IncrementalFNOTrainer with various incremental learning options.
# Other options include setting incremental_loss_gap = True.
# If one wants to use incremental resolution, set it to True.
# In this example we only update the modes and not the resolution.
# When using incremental resolution, keep in mind that the number of modes
# initially set should be strictly less than the resolution.
#
# Key parameters for incremental training:
#
# - incremental_grad: bool, default is False. If True, use the base incremental algorithm based on gradient variance
# - incremental_grad_eps: threshold for gradient variance
# - incremental_buffer: number of buffer modes to calculate gradient variance
# - incremental_max_iter: initial number of iterations
# - incremental_grad_max_iter: maximum iterations to accumulate gradients
# - incremental_loss_gap: bool, default is False. If True, use the incremental algorithm based on loss gap
# - incremental_loss_eps: threshold for loss gap

# Create the IncrementalFNOTrainer with our configuration
trainer = IncrementalFNOTrainer(
    model=model,
    n_epochs=20,
    data_processor=data_transform,
    device=device,
    verbose=True,
    incremental_loss_gap=False,  # Use gradient-based incremental learning
    incremental_grad=True,  # Enable gradient-based mode updates
    incremental_grad_eps=0.9999,  # Gradient variance threshold
    incremental_loss_eps=0.001,  # Loss gap threshold
    incremental_buffer=5,  # Buffer modes for gradient calculation
    incremental_max_iter=1,  # Initial iterations
    incremental_grad_max_iter=2,  # Maximum gradient accumulation iterations
)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Training the incremental FNO model
# ----------------------------------
# We train the model using incremental meta-learning. The trainer will:
# 1. Start with a small number of Fourier modes
# 2. Gradually increase the model capacity based on gradient variance
# 3. Monitor the incremental learning progress
# 4. Evaluate on test data throughout training

trainer.train(
    train_loader,
    test_loaders,
    optimizer,
    scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Visualizing incremental FNO predictions
# ----------------------------------------
# We visualize the model's predictions after incremental training.
# Note that we trained on a very small resolution for a very small number of epochs.
# In practice, we would train at larger resolution on many more samples.
#
# However, for practicality, we created a minimal example that:
# i) fits in just a few MB of memory
# ii) can be trained quickly on CPU
#
# In practice we would train a Neural Operator on one or multiple GPUs

test_samples = test_loaders[32].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    # Input x
    x = data["x"].to(device)
    # Ground-truth
    y = data["y"].to(device)
    # Model prediction: incremental FNO output
    out = model(x.unsqueeze(0))

    # Plot input x
    ax = fig.add_subplot(3, 3, index * 3 + 1)
    x = x.cpu().squeeze().detach().numpy()
    y = y.cpu().squeeze().detach().numpy()
    ax.imshow(x, cmap="gray")
    if index == 0:
        ax.set_title("Input x")
    plt.xticks([], [])
    plt.yticks([], [])

    # Plot ground-truth y
    ax = fig.add_subplot(3, 3, index * 3 + 2)
    ax.imshow(y.squeeze())
    if index == 0:
        ax.set_title("Ground-truth y")
    plt.xticks([], [])
    plt.yticks([], [])

    # Plot model prediction
    ax = fig.add_subplot(3, 3, index * 3 + 3)
    ax.imshow(out.cpu().squeeze().detach().numpy())
    if index == 0:
        ax.set_title("Incremental FNO prediction")
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle("Incremental FNO predictions on Darcy-Flow data", y=0.98)
plt.tight_layout()
fig.show()
