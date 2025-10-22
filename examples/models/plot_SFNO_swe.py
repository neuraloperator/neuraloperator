"""
Training a SFNO on the spherical Shallow Water equations
==========================================================

Using the small Spherical Shallow Water Equations example we ship with the package
to train a Spherical Fourier-Neural Operator (SFNO).

This tutorial demonstrates how to train neural operators on spherical domains, which is
crucial for many geophysical applications like weather prediction, ocean modeling, and
climate science. The SFNO extends the FNO architecture to handle data on the sphere
using spherical harmonics instead of regular Fourier modes.

The Shallow Water Equations describe the motion of a thin layer of fluid and are
fundamental in atmospheric and oceanic dynamics.
"""

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Import dependencies
# -------------------
# We import the necessary modules for training a Spherical Fourier Neural Operator

import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import SFNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_spherical_swe
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Loading the Spherical Shallow Water Equations dataset
# -----------------------------------------------------
# We load the spherical shallow water equations dataset with multiple resolutions.
# The dataset contains velocity and height fields on the sphere, which are the
# fundamental variables in shallow water dynamics.

train_loader, test_loaders = load_spherical_swe(
    n_train=200,
    batch_size=32,
    train_resolution=(32, 64),
    test_resolutions=[(32, 64), (64, 128)],
    n_tests=[40, 40],
    test_batch_sizes=[40, 40],
)


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Creating the Spherical FNO model
# ---------------------------------


model = SFNO(
    n_modes=(16, 32),
    in_channels=3,
    out_channels=3,
    hidden_channels=64,
    domain_padding=[0.05, 0.05],
    n_layers=2,
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
# We use AdamW optimizer with a lower learning rate for spherical data
optimizer = AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Setting up loss functions
# -------------------------
# For spherical data, we use L2 loss with sum reduction to handle the varying
# grid sizes across different latitudes on the sphere
l2loss = LpLoss(d=2, p=2, reduction="sum")

train_loss = l2loss
eval_losses = {"l2": l2loss}


# %%


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
# We create a Trainer object that handles the training loop for spherical data
trainer = Trainer(
    model=model,
    n_epochs=30,
    device=device,
    wandb_log=False,  # Disable Weights & Biases logging
    eval_interval=5,  # Evaluate every 5 epochs
    use_distributed=False,  # Single GPU/CPU training
    verbose=True,  # Print training progress
)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Training the SFNO model
# ------------------------
# We train the model on the spherical shallow water equations dataset.
# The trainer will handle the forward pass through the SFNO, compute the L2 loss,
# backpropagate, and evaluate on test data.

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
# Visualizing SFNO predictions on spherical data
# ------------------------------------------------
# We visualize the model's predictions on spherical shallow water equations data.
# Note that we trained on a very small resolution for a very small number of epochs.
# In practice, we would train at larger resolution on many more samples.
#
# However, for practicality, we created a minimal example that:
# i) fits in just a few MB of memory
# ii) can be trained quickly on CPU
#
# In practice we would train a Neural Operator on one or multiple GPUs

fig = plt.figure(figsize=(14, 7))
for index, resolution in enumerate([(32, 64), (64, 128)]):
    test_samples = test_loaders[resolution].dataset
    data = test_samples[0]
    # Input x
    x = data["x"]
    # Ground-truth
    y = data["y"][0, ...].numpy()
    # Model prediction: SFNO output
    x_in = x.unsqueeze(0).to(device)
    out = model(x_in).squeeze()[0, ...].detach().cpu().numpy()
    x = x[0, ...].detach().numpy()

    # Plot input fields
    ax = fig.add_subplot(2, 3, index * 3 + 1)
    ax.imshow(x)
    ax.set_title(f"Input x {resolution}")
    plt.xticks([], [])
    plt.yticks([], [])

    # Compute the min and max to use consistent color mapping
    vmin = y.min()
    vmax = y.max()

    # Plot ground-truth fields
    ax = fig.add_subplot(2, 3, index * 3 + 2)
    im_gt = ax.imshow(y, vmin=vmin, vmax=vmax)
    ax.set_title("Ground-truth y")
    plt.xticks([], [])
    plt.yticks([], [])

    # Plot model prediction
    ax = fig.add_subplot(2, 3, index * 3 + 3)
    im_pred = ax.imshow(out, vmin=vmin, vmax=vmax)
    ax.set_title("SFNO prediction")
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle("SFNO predictions on spherical shallow water equations", y=0.98, fontsize=24)
plt.tight_layout()
fig.show()
