"""
U-NO on Darcy-Flow
==================

Training a U-shaped Neural Operator (U-NO) on the small Darcy-Flow example we ship with the package.

This tutorial demonstrates the U-NO architecture, which combines the resolution invariance
of neural operators with the multi-scale feature extraction of U-Net architectures.
The U-NO uses skip connections and multi-resolution processing to capture both local
and global features in the data, making it particularly effective for complex PDE problems.

"""

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Import dependencies
# -------------------
# We import the necessary modules for working with the UNO model

import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import UNO
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
# We load the Darcy-Flow dataset for training and testing.

train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=32,
    n_tests=[100, 50],
    test_resolutions=[16, 32],
    test_batch_sizes=[32, 32],
)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Creating the U-NO model
# ------------------------
# We create a U-shaped Neural Operator with the following architecture:
#
# - in_channels: Number of input channels
# - out_channels: Number of output channels
# - hidden_channels: Width of the hidden layers
# - uno_out_channels: Channel dimensions for each layer in the U-Net structure
# - uno_n_modes: Fourier modes for each layer (decreasing then increasing)
# - uno_scalings: Scaling factors for each layer

model = UNO(
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    projection_channels=64,
    uno_out_channels=[32, 64, 64, 64, 32],
    uno_n_modes=[[8, 8], [8, 8], [4, 4], [8, 8], [8, 8]],
    uno_scalings=[[1.0, 1.0], [0.5, 0.5], [1, 1], [2, 2], [1, 1]],
    horizontal_skips_map=None,
    channel_mlp_skip="linear",
    n_layers=5,
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
optimizer = AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

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
# Displaying configuration
# ------------------------
# We print the model architecture, optimizer, scheduler, and loss functions

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
# ---------------------
# We create a Trainer object that handles the training loop for the U-NO
trainer = Trainer(
    model=model,
    n_epochs=30,
    device=device,
    data_processor=data_processor,
    wandb_log=False,  # Disable Weights & Biases logging
    eval_interval=5,  # Evaluate every 5 epochs
    use_distributed=False,  # Single GPU/CPU training
    verbose=True,
)  # Print training progress

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Training the U-NO model
# ------------------------
# We train the model on our Darcy-Flow dataset. The trainer will:
#
# 1. Run the forward pass through the U-NO
# 2. Compute the H1 loss
# 3. Backpropagate and update weights
# 4. Evaluate on test data every 5 epochs


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
# Visualizing U-NO predictions
# -----------------------------
# We visualize the model's predictions on the Darcy-Flow dataset.
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
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data["x"]
    # Ground-truth
    y = data["y"]
    # Model prediction: U-NO output
    out = model(x.unsqueeze(0).to(device)).cpu()

    # Plot input x
    ax = fig.add_subplot(3, 3, index * 3 + 1)
    ax.imshow(x[0], cmap="gray")
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
    ax.imshow(out.squeeze().detach().numpy())
    if index == 0:
        ax.set_title("U-NO prediction")
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle("U-NO predictions on 32x32 Darcy-Flow data", y=0.98)
plt.tight_layout()
fig.show()
