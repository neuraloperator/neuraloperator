CPU Offloading for Training Neural Operators
============================================

Enable CPU offloading to reduce GPU memory usage during training with high-resolution inputs.

Overview
--------

When training neural operators with high-resolution inputs, GPU memory usage can become a bottleneck. The peak memory consumption often exceeds CUDA limits because all intermediate activations in the computation graph are stored on the GPU by default.

Each activation tensor typically has a shape of:

.. math::

   \text{batch\_size} \times \text{hidden\_dim} \times N_x \times N_y \times \dots

where :math:`N_x, N_y, \dots` are the spatial or temporal resolutions of the input. As the computation graph grows deeper during forward and backward passes, a large number of such intermediate tensors accumulate, leading to high GPU memory consumption.

**CPU offloading** addresses this by moving activations to CPU memory during training, allowing:

- Training with higher-resolution inputs under limited GPU memory
- Training larger models without reducing batch size
- Better memory utilization across CPU and GPU

.. note::
   CPU offloading trades memory for compute time, as data transfer between CPU and GPU adds overhead.

Example Usage
-------------

Below is a complete example demonstrating CPU offloading integration with NeuralOperator training.

1. Setup and Data Loading
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import matplotlib.pyplot as plt
    import sys
    from functools import wraps
    
    from neuralop.models import FNO
    from neuralop import Trainer
    from neuralop.training import AdamW
    from neuralop.data.datasets import load_darcy_flow_small
    from neuralop.utils import count_model_params
    from neuralop import LpLoss, H1Loss

    device = 'cuda'

Load the dataset:

.. code-block:: python

    # Load Darcy flow dataset with specified resolutions
    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, 
        batch_size=32,
        test_resolutions=[16, 32], 
        n_tests=[100, 50],
        test_batch_sizes=[32, 32],
    )
    data_processor = data_processor.to(device)

2. Model Creation
^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create FNO model with specified parameters
    model = FNO(
        n_modes=(16, 16),           # Fourier modes for each dimension
        in_channels=1,              # Input channels
        out_channels=1,             # Output channels  
        hidden_channels=32,         # Hidden layer width
        projection_channel_ratio=2  # Channel expansion ratio
    )
    model = model.to(device)
    
    print(f"Model parameters: {count_model_params(model)}")

3. Enable CPU Offloading
^^^^^^^^^^^^^^^^^^^^^^^^

Wrap the model's forward function to enable automatic CPU offloading:

.. code-block:: python

    def wrap_forward_with_offload(forward_fn):
        """
        Wrap a forward function to enable CPU offloading of activations.
        
        Parameters
        ----------
        forward_fn : callable
            The original forward function to wrap
            
        Returns
        -------
        callable
            Wrapped forward function with CPU offloading enabled
        """
        @wraps(forward_fn)
        def wrapped_forward(*args, **kwargs):
            # Enable CPU offloading context for this forward pass
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                return forward_fn(*args, **kwargs)
        return wrapped_forward

    # Apply CPU offloading to the model
    model.forward = wrap_forward_with_offload(model.forward)

4. Training Loop
^^^^^^^^^^^^^^^^

No changes are needed in your existing training code:

.. code-block:: python

    # Setup optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    
    # Training step - works exactly as before
    for batch_idx, (input_data, target_data) in enumerate(train_loader):
        # Move data to device
        input_data = input_data.to(device)    # Shape: (batch, channels, height, width)
        target_data = target_data.to(device)  # Shape: (batch, channels, height, width)
        
        # Forward pass - activations automatically offloaded to CPU
        output = model(input_data)
        
        # Compute loss
        loss = l2loss(output, target_data)
        
        # Backward pass - gradients computed with CPU-stored activations
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

Performance Considerations
-------------------------

**Memory vs Speed Trade-off**
    CPU offloading reduces GPU memory usage at the cost of increased training time due to data transfer overhead between CPU and GPU memory.

**When to Use**
    - Training fails with CUDA out-of-memory errors
    - You want to increase batch size or model resolution
    - GPU memory is the primary bottleneck

**When Not to Use**  
    - GPU memory is sufficient for your current setup
    - Training speed is more critical than memory usage
    - CPU memory is also limited

**Optimization Tips**
    - Use ``pin_memory=True`` for faster CPU-GPU transfers
    - Consider gradient checkpointing as an alternative memory-saving technique
    - Monitor both GPU and CPU memory usage during training

.. warning::
   CPU offloading requires PyTorch version 1.12.0 or higher. Ensure your environment meets this requirement before using this feature.