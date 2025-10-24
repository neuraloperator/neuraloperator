"""
Training an FNO on Darcy-Flow
=============================

We train a Fourier Neural Operator (FNO) on our small :ref:`Darcy-Flow example <sphx_glr_auto_examples_data_plot_darcy_flow.py>`.

This tutorial demonstrates the complete workflow of training a neural operator:
1. Loading and preprocessing the Darcy-Flow dataset
2. Creating an FNO model architecture
3. Setting up training components (optimizer, scheduler, losses)
4. Training the model
5. Evaluating predictions and zero-shot super-resolution

Note that this dataset is much smaller than one we would use in practice. The small Darcy-flow is an example built to
be trained on a CPU in a few seconds, whereas normally we would train on one or multiple GPUs. 

The FNO's key advantage is its resolution invariance - it can make predictions at different resolutions
without retraining, which we will demonstrate in the zero-shot super-resolution section.
"""

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Import dependencies
# -------------------
# We import the necessary modules from `neuralop` for training a Fourier Neural Operator

import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

device = "cpu"

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Loading the Darcy-Flow dataset
# ------------------------------
# We load the small Darcy-Flow dataset with multiple resolutions for training and testing.
# The dataset contains permeability fields (input) and pressure fields (output).

train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=64,
    n_tests=[100, 50],
    test_resolutions=[16, 32],
    test_batch_sizes=[32, 32],
)
data_processor = data_processor.to(device)


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Creating the FNO model
# ----------------------

model = FNO(
    n_modes=(8, 8),
    in_channels=1,
    out_channels=1,
    hidden_channels=24,
    projection_channel_ratio=2,
)
model = model.to(device)

# Count and display the number of parameters
n_params = count_model_params(model)
print(f"\nOur model has {n_params} parameters.")
sys.stdout.flush()


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Creating the optimizer and scheduler
# ------------------------------------
# We use AdamW optimizer with weight decay for regularization
optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Setting up loss functions
# -------------------------
# We use H1 loss for training and L2 loss for evaluation
# H1 loss is particularly good for PDE problems as it penalizes both function values and gradients
l2loss = LpLoss(d=2, p=2)  # L2 loss for function values
h1loss = H1Loss(d=2)  # H1 loss includes gradient information

train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Training the model
# ---------------------
# We display the training configuration and then train the model

print("\n### MODEL ###\n", model)
print("\n### OPTIMIZER ###\n", optimizer)
print("\n### SCHEDULER ###\n", scheduler)
print("\n### LOSSES ###")
print(f"\n * Train: {train_loss}")
print(f"\n * Test: {eval_losses}")
sys.stdout.flush()

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Creating the trainer
# --------------------
# We create a Trainer object that handles the training loop, evaluation, and logging
trainer = Trainer(
    model=model,
    n_epochs=15,
    device=device,
    data_processor=data_processor,
    wandb_log=False,  # Disable Weights & Biases logging for this tutorial
    eval_interval=5,  # Evaluate every 5 epochs
    use_distributed=False,  # Single GPU/CPU training
    verbose=True,  # Print training progress
)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Training the model
# ------------------
# We train the model on our Darcy-Flow dataset. The trainer will:
# 1. Run the forward pass through the FNO
# 2. Compute the H1 loss
# 3. Backpropagate and update weights
# 4. Evaluate on test data every 3 epochs

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# .. _plot_preds :
# Visualizing predictions
# ------------------------
# Let's take a look at what our model's predicted outputs look like.
# We wll compare the inputs, ground-truth outputs, and model predictions side by side.
#
# Note that in this example, we train on a very small resolution for
# a very small number of epochs. In practice, we would train at a larger
# resolution on many more samples.

test_samples = test_loaders[16].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)

    # Input
    x = data["x"]
    # Ground-truth output
    y = data["y"]
    # Model prediction
    out = model(x.unsqueeze(0))

    # Plot input
    ax = fig.add_subplot(3, 3, index * 3 + 1)
    ax.imshow(x[0], cmap="gray")
    if index == 0:
        ax.set_title("Input x")
    plt.xticks([], [])
    plt.yticks([], [])

    # Plot ground-truth output
    ax = fig.add_subplot(3, 3, index * 3 + 2)
    ax.imshow(y.squeeze())
    if index == 0:
        ax.set_title("Ground-truth output")
    plt.xticks([], [])
    plt.yticks([], [])

    # Plot model prediction
    ax = fig.add_subplot(3, 3, index * 3 + 3)
    ax.imshow(out.squeeze().detach().numpy())
    if index == 0:
        ax.set_title("Model prediction")
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle("FNO predictions on 16x16 Darcy-Flow data", y=0.98)
plt.tight_layout()
fig.show()


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# .. zero_shot :
# Zero-shot super-resolution evaluation
# -------------------------------------
# One of the key advantages of neural operators is their resolution invariance.
# The FNO's invariance to the discretization of input data means we can natively
# make predictions on higher-resolution inputs and get higher-resolution outputs
# without retraining the model!

test_samples = test_loaders[32].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)

    # Input at higher-resolution
    x = data["x"]
    # Ground-truth output at higher-resolution
    y = data["y"]
    # Model prediction at higher-resolution
    out = model(x.unsqueeze(0))

    # Plot input at higher-resolution
    ax = fig.add_subplot(3, 3, index * 3 + 1)
    ax.imshow(x[0], cmap="gray")
    if index == 0:
        ax.set_title("Input at 32x32")
    plt.xticks([], [])
    plt.yticks([], [])

    # Plot ground-truth output at higher-resolution
    ax = fig.add_subplot(3, 3, index * 3 + 2)
    ax.imshow(y.squeeze())
    if index == 0:
        ax.set_title("Ground-truth at 32x32")
    plt.xticks([], [])
    plt.yticks([], [])

    # Plot model prediction at higher-resolution
    ax = fig.add_subplot(3, 3, index * 3 + 3)
    ax.imshow(out.squeeze().detach().numpy())
    if index == 0:
        ax.set_title("FNO prediction at 32x32")
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle("Zero-shot super-resolution: 16x16 → 32x32", y=0.98)
plt.tight_layout()
fig.show()

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Understanding zero-shot super-resolution
# ----------------------------------------
# We only trained the model on data at a resolution of 16x16, and with no modifications
# or special prompting, we were able to perform inference on higher-resolution input data
# and get higher-resolution predictions! This is a powerful capability of neural operators.
#
# In practice, we often want to evaluate neural operators at multiple resolutions to track
# a model's zero-shot super-resolution performance throughout training. That's why many of
# our datasets, including the small Darcy-flow we showcased, are parameterized with a list
# of `test_resolutions` to choose from.
#
# Note: These predictions may be noisier than we would expect for a model evaluated
# at the same resolution at which it was trained. This is because the model hasn't seen
# the higher-frequency patterns present in the 32x32 data during training. However, this
# demonstrates the fundamental resolution invariance of neural operators.
