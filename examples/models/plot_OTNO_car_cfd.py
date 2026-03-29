"""
Training OTNO on a Car CFD Dataset
==========================================

We load a pre-generated optimal transport (OT) dataset from car CFD data and train an OTNO model on it.

This tutorial demonstrates how to:

1. Load and preprocess optimal transport data for car CFD simulations
2. Create and configure an OTNO model for pressure field prediction
3. Train the model using the Trainer with proper normalization
4. Visualize the input OT maps, ground truth, and model predictions in 3D

"""

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Imports and setup
# ------------------------------
from copy import deepcopy
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from neuralop.models import OTNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_saved_ot, CFDDataProcessor
from neuralop.utils import count_model_params
from neuralop import LpLoss

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Loading the Car OT dataset
# ------------------------------
# We load the small Car OT dataset.
# The dataset contains OT maps (input) and pressure fields (output).
data_module = load_saved_ot(
    n_train=2,
    n_test=1,
    expand_factor=3.0,
    reg=1e-06,
)

train_loader = data_module.train_loader(batch_size=1, shuffle=True)
test_loader = data_module.test_loader(batch_size=1, shuffle=False)

output_encoder = deepcopy(data_module.normalizers["press"])
data_processor = CFDDataProcessor(normalizer=output_encoder)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Creating the OTNO model
# ----------------------
model = OTNO(
    n_modes=(16, 16),
    hidden_channels=64,
    in_channels=9,
    out_channels=1,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    norm="group_norm",
    use_channel_mlp=True,
    channel_mlp_expansion=1.0,
)

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
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Setting up loss functions
# -------------------------
# We use L2 loss for training and evaluation
l2loss = LpLoss(d=2, p=2)
train_loss_fn = l2loss
test_loss_fn = l2loss

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
print(f"\n * Train: {train_loss_fn}")
print(f"\n * Test: {test_loss_fn}")
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
    data_processor=data_processor,
    wandb_log=False,  # Disable Weights & Biases logging for this tutorial
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
# We train the model on our car cfd dataset. The trainer will:

# 1. Run the forward pass through the OTNO
# 2. Compute the L2 loss
# 3. Backpropagate and update weights
# 4. Evaluate on test data

trainer.train(
    train_loader=train_loader,
    test_loaders={"": test_loader},
    optimizer=optimizer,
    scheduler=scheduler,
    training_loss=train_loss_fn,
    eval_losses={"l2": test_loss_fn},
    regularizer=None,
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
# We will compare the inputs, ground-truth outputs, and model predictions side by side.
#
# Note that in this example, we train on 2 cars and test on 1 car. In practice, you would train on a larger
# number of cars for better generalization.

test_sample = test_loader.dataset[0]

# Preprocess the sample
model.eval()
with torch.no_grad():
    # Preprocess the data
    processed_sample = data_processor.preprocess(test_sample.copy())

    # Get model prediction
    x = processed_sample["x"].unsqueeze(0)  # Add batch dimension
    ind_dec = processed_sample["ind_dec"]

    # Forward pass
    prediction = model(x, ind_dec)

    # Inverse transform to get actual pressure values
    prediction = output_encoder.inverse_transform(prediction.reshape(-1, 1))
    ground_truth = output_encoder.inverse_transform(
        processed_sample["y"].reshape(-1, 1)
    )

# Extract geometry data
vertices = test_sample["target"].numpy()  # Target mesh vertices
source = test_sample["source"].numpy()  # Source mesh vertices
ind_enc = test_sample["ind_enc"].numpy()  # Encoder indices
trans = vertices[ind_enc, :]  # Transport coordinates

# Calculate axis limits for equal aspect ratio
x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
mid_x, mid_y, mid_z = (x.max() + x.min()) * 0.5, (y.max() + y.min()) * 0.5, (z.max() + z.min()) * 0.5

# Set common color scale for pressure plots
vmin = min(ground_truth.min().item(), prediction.min().item())
vmax = max(ground_truth.max().item(), prediction.max().item())

# Create RGB colors from transport coordinates
color_x = (trans[:, 0] - trans[:, 0].min()) / (trans[:, 0].max() - trans[:, 0].min())
color_y = (trans[:, 1] - trans[:, 1].min()) / (trans[:, 1].max() - trans[:, 1].min())
color_z = (trans[:, 2] - trans[:, 2].min()) / (trans[:, 2].max() - trans[:, 2].min())
colors = np.stack([color_x, color_y, color_z], axis=1)

# Create three-panel visualization
fig = plt.figure(figsize=(18, 6))

# Panel 1: Input OT with RGB-colored transport
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
scatter1 = ax1.scatter(source[:, 0], source[:, 1], source[:, 2], c=colors, alpha=0.5, s=15)
ax1.set_xlim(mid_x - max_range, mid_x + max_range)
ax1.set_ylim(mid_y - max_range, mid_y + max_range)
ax1.set_zlim(mid_z - max_range, mid_z + max_range)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.set_title("Input OT\n(RGB: Transport coordinates)", fontsize=10)
ax1.view_init(elev=20, azim=150, roll=0, vertical_axis="y")
ax1.text2D(
    0.05,
    0.95,
    "Color = RGB(trans_x, trans_y, trans_z)",
    transform=ax1.transAxes,
    fontsize=8,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

# Panel 2: Ground truth pressure
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
scatter2 = ax2.scatter(x, y, z, s=2, c=ground_truth.cpu().numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
ax2.set_xlim(mid_x - max_range, mid_x + max_range)
ax2.set_ylim(mid_y - max_range, mid_y + max_range)
ax2.set_zlim(mid_z - max_range, mid_z + max_range)
ax2.set_box_aspect([1, 1, 1])
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")
ax2.set_title("Ground Truth Pressure")
ax2.view_init(elev=20, azim=150, roll=0, vertical_axis="y")

# Panel 3: Model prediction
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
scatter3 = ax3.scatter(x, y, z, s=2, c=prediction.cpu().numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
ax3.set_xlim(mid_x - max_range, mid_x + max_range)
ax3.set_ylim(mid_y - max_range, mid_y + max_range)
ax3.set_zlim(mid_z - max_range, mid_z + max_range)
ax3.set_box_aspect([1, 1, 1])  # Force equal box aspect
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("z")
ax3.set_title("Model Prediction")
ax3.view_init(elev=20, azim=150, roll=0, vertical_axis="y")

# Add a colorbar
fig.colorbar(scatter2, ax=[ax2, ax3], pad=0.1, label="Pressure", shrink=0.8)

plt.show()

# Print error statistics
print(f"\n### Prediction Statistics ###")
print(
    f"Relative L2 Error: {(torch.norm(prediction - ground_truth) / torch.norm(ground_truth)).item():.6f}"
)

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Visualizing OT encoding and decoding
# -------------------------------------
# We display the optimal transport process as animations showing how the
# car surface is mapped to the latent torus grid and back.

pressure_pullback = ground_truth[ind_enc].numpy()
n_s = source.shape[0]

# OT encoding from the car surface to the latent torus grid
T = 60
movement_enc = np.zeros((T, n_s, 3))

for j in range(n_s):
    # Animate from CAR (trans) -> TORUS (source) for encoding
    tx = np.linspace(trans[j, 0], source[j, 0], T).reshape((T, 1))
    ty = np.linspace(trans[j, 1], source[j, 1], T).reshape((T, 1))
    tz = np.linspace(trans[j, 2], source[j, 2], T).reshape((T, 1))
    movement_enc[:, j, :] = np.concatenate((tx, ty, tz), axis=1)

# Create a Matplotlib 3D animation for the encoding (trans -> source)
print("Creating car to torus animation (matplotlib)...")
fig_enc = plt.figure(figsize=(5, 5))
ax_enc = fig_enc.add_subplot(111, projection="3d")
sc_enc = ax_enc.scatter(
    movement_enc[0, :, 0],
    movement_enc[0, :, 1],
    movement_enc[0, :, 2],
    c="grey",
    s=2,
    alpha=0.95,
    edgecolors="#7fb0b6",
    linewidths=0.03,
    depthshade=True,
)
ax_enc.set_xlim(mid_x - max_range, mid_x + max_range)
ax_enc.set_ylim(mid_y - max_range, mid_y + max_range)
ax_enc.set_zlim(mid_z - max_range, mid_z + max_range)
ax_enc.set_title("OT Encoding: Car → Torus")
ax_enc.view_init(elev=20, azim=150, roll=0, vertical_axis="y")


def update_enc(frame):
    xs = movement_enc[frame, :, 0]
    ys = movement_enc[frame, :, 1]
    zs = movement_enc[frame, :, 2]
    sc_enc._offsets3d = (xs, ys, zs)
    ax_enc.set_title(f"OT Encoding: frame {frame}")
    return (sc_enc,)


ani_enc = animation.FuncAnimation(
    fig_enc, update_enc, frames=T, interval=50, blit=False
)

# OT decoding process from the latent torus grid to the car surface
T = 60
movement_dec = np.zeros((T, n_s, 3))

for j in range(n_s):
    # Animate from TORUS (source) -> CAR (trans) for decoding
    tx = np.linspace(source[j, 0], trans[j, 0], T).reshape((T, 1))
    ty = np.linspace(source[j, 1], trans[j, 1], T).reshape((T, 1))
    tz = np.linspace(source[j, 2], trans[j, 2], T).reshape((T, 1))
    movement_dec[:, j, :] = np.concatenate((tx, ty, tz), axis=1)

print("Creating torus to car animation (matplotlib) with pressure...")
fig_dec = plt.figure(figsize=(8, 6))
ax_dec = fig_dec.add_subplot(111, projection="3d")
# Initial positions: movement_dec[0] (source positions)
sc_dec = ax_dec.scatter(
    movement_dec[0, :, 0],
    movement_dec[0, :, 1],
    movement_dec[0, :, 2],
    c=pressure_pullback,
    cmap="viridis",
    s=2,
    vmin=vmin,
    vmax=vmax,
    alpha=0.95,
    edgecolors="none",
    linewidths=0.03,
    depthshade=True,
)
ax_dec.set_xlim(mid_x - max_range, mid_x + max_range)
ax_dec.set_ylim(mid_y - max_range, mid_y + max_range)
ax_dec.set_zlim(mid_z - max_range, mid_z + max_range)
ax_dec.set_title("OT Decoding: Torus → Car (pressure)")
ax_dec.view_init(elev=20, azim=150, roll=0, vertical_axis="y")


def update_dec(frame):
    xs = movement_dec[frame, :, 0]
    ys = movement_dec[frame, :, 1]
    zs = movement_dec[frame, :, 2]
    sc_dec._offsets3d = (xs, ys, zs)
    ax_dec.set_title(f"OT Decoding: frame {frame}")
    return (sc_dec,)


ani_dec = animation.FuncAnimation(
    fig_dec, update_dec, frames=T, interval=50, blit=False
)
