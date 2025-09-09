.. _training_nos:
================================
Training neural operator models
================================

Our library makes it easy for anyone with data drawn from a system governed by a PDE to train and test Neural Operator models. 
This page details the library's Python interface for training and evaluation of NOs.


The Trainer class
==================
Most users will train neural operator models on their own data in very similar ways, 
using a very standard machine learning training loop. To speed up this process, we 
provide a :code:`Trainer` class that automates much of this boilerplate logic. 
Things like loading a model to device, zeroing gradients and computing most loss 
functions are taken care of.

The :code:`Trainer` implements training in a modular fashion, meaning that more domain-specific logic 
can easily be implemented. For more specific documentation, check the API reference.

Distributed Training
=====================
We also provide a simple way to use PyTorch's :code:`DistributedDataParallel`
functionality to hold data across multiple GPUs. We use PyTorch's :code:`torchrun` elastic launcher,
so all you need to do on a multi-GPU system is the following:

::
    
    torchrun --standalone --nproc_per_node <NUM_GPUS> script.py

You may need to adjust the batch size, model parallel size and world size in 
accordance with your specific use case. See the `torchrun documentation <https://pytorch.org/docs/stable/elastic/run.html>`_ for more details.

CPU Offloading
==============

For training with high-resolution inputs that exceed GPU memory limits, NeuralOperator supports CPU offloading of activations. This technique allows training larger models or higher-resolution problems by temporarily storing intermediate computations on CPU memory.

.. toctree::
   :maxdepth: 1

   cpuoffloading
