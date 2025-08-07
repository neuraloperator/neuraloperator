CPU Offloading for Training Neural Operators
============================================

Enable CPU offloading to reduce GPU memory usage during training.

Overview
--------

When training with high-resolution inputs, the peak GPU memory usage can exceed the CUDA limit, since all activations in the computation graph are saved on the GPU by default. Each activation tensor typically has a shape of:

.. math::

   \text{batch\_size} \times \text{hidden\_dim} \times N_x \times N_y \times \dots,

where :math:`N_x, N_y, \dots` are the spatial or temporal resolutions of the input. As the computation graph grows deeper, a large number of such intermediate tensors are stored, leading to high GPU memory consumption.

To address this, we can use **CPU offloading**, which moves activations to the CPU during training to reduce GPU memory usage. This allows training with higher-resolution inputs or larger models under limited GPU memory.

Example Usage
-------------

Below is a step-by-step example of enabling CPU offloading during model training.

1. Create the model
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import matplotlib.pyplot as plt
    import sys
    from neuralop.models import FNO
    from neuralop import Trainer
    from neuralop.training import AdamW
    from neuralop.data.datasets import load_darcy_flow_small
    from neuralop.utils import count_model_params
    from neuralop import LpLoss, H1Loss

    device = 'cuda'

Load the dataset:

.. code-block:: python

    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=32,
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
    )
    data_processor = data_processor.to(device)

Create the model:

.. code-block:: python

    model = FNO(n_modes=(16, 16),
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                projection_channel_ratio=2)
    model = model.to(device)

2. Wrap the forward function of the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To enable CPU offloading of activations:

.. code-block:: python

    from functools import wraps

    def wrap_forward_with_offload(forward_fn):
        @wraps(forward_fn)
        def wrapped_forward(*args, **kwargs):
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                return forward_fn(*args, **kwargs)
        return wrapped_forward

    model.forward = wrap_forward_with_offload(model.forward)

3. Call the model during training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No changes are needed in your training loop.

Forward pass:

.. code-block:: python

    output = model(input)

Compute the training loss:

.. code-block:: python

    loss = compute_loss(output, target)

Backward pass:

.. code-block:: python

    loss.backward()
