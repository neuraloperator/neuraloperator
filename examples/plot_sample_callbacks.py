"""
Callbacks in the Trainer module
=============================

In this example, we demonstrate how to use Callbacks to configure
domain-specific logic within the training loop of a Trainer.
"""

# %%
# 
import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.training import Callback
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

device = 'cpu'


# %%
"""
Let's set up the Trainer as before to train a TFNO on the small Darcy Flow dataset:
"""
# Loading the Navier-Stokes dataset in 128x128 resolution
train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=300, batch_size=32, 
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
)

# %%
# Create the model, optimizer, scheduler, and loss functions as before
model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}

# %%
# Now, let's look at Callbacks in more depth. The Trainer class takes a list of Callback objects at creation time. These Callbacks allow you to keep track of the training state and implement more domain-specific behavior within the Trainer's automated loops.

# %%
# Let's start with a simple Callback that just prints to stdout at the beginning and end of each training epoch.

# %% 
class EpochPrintCallback(Callback):
    """EpochPrintCallback is a simple callback that 
       just prints out to the stdout at the start of each epoch.
    """
    def __init__(self):
        super().__init__()

    def on_epoch_start(self, epoch, *args, **kwargs):
        self._update_state_dict(epoch=epoch) # update the Callback's internal state
        print(f"Starting epoch {epoch}")
        return
    
    def on_epoch_end(self, *args, **kwargs):
        # the epoch persists as part of the Callback's state
        print(f"Finishing epoch {self.state_dict['epoch']}")
        return

# Create the trainer
trainer = Trainer(model=model, n_epochs=5,
                  device=device,
                  callbacks=[
                    EpochPrintCallback()
                        ],
                  data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=1,
                  use_distributed=False,
                  verbose=True)


# %%
# Let's train the model and see what our callback outputs.

trainer.train(train_loader=train_loader,
              test_loaders={},
              
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss)

# %%
# We did that without modifying the trainer code at all! Callbacks also have access to the entire training state (model, optimizer, scheduler) through the arguments passed to them over the course of training. This enables all sorts of user-defined complex behavior. Take a look at the Incremental-FNO example for an example!