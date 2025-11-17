"""
.. _small_darcy_vis :

A simple Darcy-Flow dataset
===========================
An introduction to the small Darcy-Flow example dataset we ship with the package.

The Darcy-Flow problem is a fundamental partial differential equation (PDE) in fluid mechanics
that describes the flow of a fluid through a porous medium. In this tutorial, we explore the
dataset structure and visualize how the data is processed for neural operator training.

"""

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Import the library
# ------------------
# We first import our `neuralop` library and required dependencies.

import matplotlib.pyplot as plt
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.layers.embeddings import GridEmbedding2D

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Load the dataset
# ----------------
# Training samples are 16x16 and we load testing samples at both
# 16x16 and 32x32 (to test resolution invariance).

train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=20,
    batch_size=4,
    test_resolutions=[16, 32],
    n_tests=[10, 10],
    test_batch_sizes=[4, 2],
)

train_dataset = train_loader.dataset

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Visualizing the data
# --------------------
# Let's examine the shape and structure of our dataset at different resolutions.

for res, test_loader in test_loaders.items():
    print(f"Resolution: {res}")
    # Get first batch
    batch = next(iter(test_loader))
    x = batch["x"]  # Input
    y = batch["y"]  # Output

    print(f"Testing samples for resolution {res} have shape {x.shape[1:]}")


data = train_dataset[0]
x = data["x"]
y = data["y"]

print(f"Training samples have shape {x.shape[1:]}")

# Which sample to view
index = 0

data = train_dataset[index]
data = data_processor.preprocess(data, batched=False)

# The first step of the default FNO model is a grid-based
# positional embedding. We will add it manually here to
# visualize the channels appended by this embedding.
positional_embedding = GridEmbedding2D(in_channels=1)
# At train time, data will be collated with a batch dimension.
# We create a batch dimension to pass into the embedding, then re-squeeze
x = positional_embedding(data["x"].unsqueeze(0)).squeeze(0)
y = data["y"]

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Visualizing the processed data
# ------------------------------
# We can see how the positional embedding adds coordinate information to our input data.
# This helps the neural operator understand spatial relationships in the data.

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(2, 2, 1)
ax.imshow(x[0], cmap="gray")
ax.set_title("Input x")
ax = fig.add_subplot(2, 2, 2)
ax.imshow(y.squeeze())
ax.set_title("Output y")
ax = fig.add_subplot(2, 2, 3)
ax.imshow(x[1])
ax.set_title("Positional embedding: x-coordinates")
ax = fig.add_subplot(2, 2, 4)
ax.imshow(x[2])
ax.set_title("Positional embedding: y-coordinates")
fig.suptitle("Visualizing one input sample with positional embeddings", y=0.98)
plt.tight_layout()
fig.show()
