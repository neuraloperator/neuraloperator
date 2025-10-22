.. _user_guide :
User Guide
===========

NeuralOperator provides all the tools you need 
to easily use, build and train neural operators for your own applications
and learn mapping between function spaces, in PyTorch.


NeuralOperator library structure
---------------------------------

Here are the main components of the library:

================================= ================================
Module                             Description
================================= ================================
:mod:`neuralop`                   Main library with core imports
:mod:`neuralop.models`            Full ready-to-use neural operators (FNO, SFNO, UNO, UQNO, FNOGNO, GINO, etc.)
:mod:`neuralop.layers`            Individual layers to build neural operators
:mod:`neuralop.data`              Convenience PyTorch data loaders for PDE datasets and transforms
:mod:`neuralop.training`          Utilities to train neural operators end-to-end (Trainer, AdamW, etc.)
:mod:`neuralop.losses`            Loss functions for neural operator training (LpLoss, H1Loss, etc.)
:mod:`neuralop.mpu`               Multi-processing utilities for distributed training
================================= ================================

The main :mod:`neuralop` module provides convenient imports for the most commonly used components:

- **Models**: FNO, SFNO, UNO, UQNO, FNOGNO, GINO, LocalNO, CODANO, get_model, etc...
- **Training**: Trainer
- **Losses**: LpLoss, H1Loss, WeightedSumLoss, Relobralo, SoftAdapt, FourierDiff, non_uniform_fd, FiniteDiff
- **Data**: datasets, transforms
- **Utilities**: mpu



.. raw:: html

   <div style="margin-top: 4em;"></div>

Available Neural Operator Models
---------------------------------

The :mod:`neuralop.models` module includes several state-of-the-art neural operator architectures:

- **FNO (Fourier Neural Operator)**: The original Fourier-based neural operator (1D, 2D, 3D variants)
- **TFNO (Tensorized FNO)**: Tensorized version with Tucker factorization (1D, 2D, 3D variants)
- **SFNO (Spherical FNO)**: Spherical harmonics-based FNO for spherical domains (requires torch_harmonics)
- **UNO (U-shaped Neural Operator)**: U-Net inspired architecture for neural operators
- **UQNO (Uncertainty Quantification NO)**: Neural operator with uncertainty quantification
- **FNOGNO (FNO + Graph Neural Operator)**: Hybrid FNO-GNO architecture
- **GINO (Graph Neural Operator)**: Graph-based neural operator for irregular domains
- **LocalNO**: Local neural operator for efficient computation (requires torch_harmonics)
- **CODANO**: Continuous-discrete neural operator


.. raw:: html

   <div style="margin-top: 4em;"></div>

Data Loading and Preprocessing
------------------------------

The :mod:`neuralop.data` module provides comprehensive data handling capabilities:

**Datasets** (:mod:`neuralop.data.datasets`):

- **Darcy Flow**: Standard benchmark for elliptic PDEs (load_darcy_flow_small, load_darcy_pt)
- **Burgers Equation**: Nonlinear PDE benchmark (load_mini_burgers_1dtime)
- **Navier-Stokes**: Fluid dynamics equations (load_navier_stokes_pt)
- **Spherical SWE**: Shallow water equations on spherical domains (load_spherical_swe, requires torch_harmonics)
- **Car CFD**: Computational fluid dynamics data (load_mini_car)
- **Nonlinear Poisson**: Poisson equation with nonlinear terms (load_nonlinear_poisson_pt)
- **The Well**: Active matter and MHD datasets (requires the_well package)

**Transforms** (:mod:`neuralop.data.transforms`):

- **Normalizers**: UnitGaussianNormalizer, DictUnitGaussianNormalizer
- **Data Processors**: DefaultDataProcessor, IncrementalDataProcessor, MGPatchingDataProcessor
- **Patching Transforms**: For handling large-scale problems
- **Base Transforms**: Extensible Transform and DictTransform framework

.. raw:: html

   <div style="margin-top: 4em;"></div>

Training Neural Operator Models
--------------------------------

Our library makes it easy for anyone with data drawn from a system governed by a 
PDE to train and test Neural Operator models. 
The library provides comprehensive training utilities and loss functions to 
get you started quickly.

.. raw:: html

   <div style="margin-top: 2em;"></div>

The Trainer Class
~~~~~~~~~~~~~~~~~

Most users will train neural operator models on their own data in very similar 
ways, using a very standard machine learning training loop. 
To speed up this process, we provide a :code:`Trainer` class that automates 
much of this boilerplate logic. 
Things like loading a model to device, zeroing gradients and computing most 
loss functions are taken care of.

The :code:`Trainer` implements training in a modular fashion, meaning that 
more domain-specific logic can easily be implemented. For more specific 
documentation, check the :ref:`api_ref`.

.. raw:: html

   <div style="margin-top: 2em;"></div>

Available Training Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :mod:`neuralop.training` module provides several key components:

- **Trainer**: Main training class for neural operator models
- **AdamW**: Optimized Adam optimizer with weight decay
- **IncrementalFNOTrainer**: Specialized trainer for incremental FNO training
- **setup**: PyTorch setup utilities for distributed training
- **load_training_state/save_training_state**: Utilities for checkpointing

Note: The main :mod:`neuralop` module directly imports `Trainer` for convenience.

.. raw:: html

   <div style="margin-top: 2em;"></div>

Loss Functions
~~~~~~~~~~~~~~

The :mod:`neuralop.losses` module provides various loss functions:

- **Data Losses**: LpLoss, H1Loss for standard regression tasks
- **Equation Losses**: Various equation-specific loss functions for physics-informed training
- **Meta Losses**: WeightedSumLoss, Aggregator, Relobralo, SoftAdapt for advanced training strategies
- **Differentiation**: FourierDiff, non_uniform_fd, FiniteDiff for computing derivatives

.. raw:: html

   <div style="margin-top: 2em;"></div>

Distributed Training
~~~~~~~~~~~~~~~~~~~~

We also provide a simple way to use PyTorch's :code:`DistributedDataParallel` functionality 
to hold data across multiple GPUs. 
We use PyTorch's :code:`torchrun` elastic launcher, so all you need to do on a 
multi-GPU system is the following:

::
    
    torchrun --standalone --nproc_per_node <NUM_GPUS> script.py

You may need to adjust the batch size, model parallel size and world size 
in accordance with your specific use case. 
See the `torchrun documentation <https://pytorch.org/docs/stable/elastic/run.html>`_ 
for more details.

.. raw:: html

   <div style="margin-top: 4em;"></div>

CPU Offloading
--------------

For training with high-resolution inputs that exceed GPU memory limits, 
NeuralOperator supports CPU offloading of activations. 
This technique allows training larger models or higher-resolution problems 
by temporarily storing intermediate computations on CPU memory.

.. raw:: html

   <div style="margin-top: 2em;"></div>

Overview
~~~~~~~~

When training neural operators with high-resolution inputs, GPU memory 
usage can become a bottleneck. 
The peak memory consumption often exceeds CUDA limits because all 
intermediate activations in the computation graph are stored on the GPU by default.

Each activation tensor typically has a shape of:

.. math::

   \text{batch_size} \times \text{hidden_dim} \times N_x \times N_y \times \dots

where :math:`N_x, N_y, \dots` are the spatial or temporal resolutions of the input.
As the computation graph grows deeper during forward and backward passes, 
a large number of such intermediate tensors accumulate, leading 
to high GPU memory consumption.

**CPU offloading** addresses this by moving activations to CPU memory 
during training, allowing:

- Training with higher-resolution inputs under limited GPU memory
- Training larger models without reducing batch size
- Better memory utilization across CPU and GPU

.. note::
   CPU offloading trades memory for compute time, as data transfer between 
   CPU and GPU adds overhead.

.. raw:: html

   <div style="margin-top: 2em;"></div>

Example Usage
~~~~~~~~~~~~~~

Below is a complete example demonstrating CPU offloading integration 
with NeuralOperator training:

1. Setup and Data Loading
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import matplotlib.pyplot as plt
    import sys
    from functools import wraps
    
    from neuralop.models import FNO
    from neuralop import Trainer, LpLoss, H1Loss
    from neuralop.training import AdamW
    from neuralop.data.datasets import load_darcy_flow_small
    from neuralop.utils import count_model_params

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

.. raw:: html

   <div style="margin-top: 2em;"></div>

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Memory vs Speed Trade-off**
    CPU offloading reduces GPU memory usage at the cost of increased 
    training time due to data transfer overhead between CPU and GPU memory.

.. raw:: html

   <div style="margin-top: 2em;"></div>

**When to Use**
    - Training fails with CUDA out-of-memory errors
    - You want to increase batch size or model resolution
    - GPU memory is the primary bottleneck

.. raw:: html

   <div style="margin-top: 2em;"></div>

**When Not to Use**  
    - GPU memory is sufficient for your current setup
    - Training speed is more critical than memory usage
    - CPU memory is also limited

.. raw:: html

   <div style="margin-top: 2em;"></div>

**Optimization Tips**
    - Use ``pin_memory=True`` for faster CPU-GPU transfers
    - Consider gradient checkpointing as an alternative memory-saving technique
    - Monitor both GPU and CPU memory usage during training

.. raw:: html

   <div style="margin-top: 2em;"></div>
   
.. warning::
   CPU offloading requires PyTorch version 1.12.0 or higher. Ensure your environment meets this requirement before using this feature.



.. raw:: html

   <div style="margin-top: 4em;"></div>

Interactive examples with code
-----------------------------
We also provide interactive examples that show our library and neural operator 
models in action. 
To get up to speed on the code, and look through some interactive examples 
to help you hit the ground running, check out our :ref:`gallery_examples`.

We also provide training recipe scripts for our models on sample problems 
in the `scripts` directory.

