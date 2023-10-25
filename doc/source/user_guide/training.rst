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

If you need to implement more domain-specific training logic, we also expose an
interface for stateful Callbacks that trigger events and track information
throughout the lifetime of your Trainer. Each callback implements a series of 
methods that are automatically called throughout the training loop, and you
can override individual methods as necessary for your custom logic. 

For more specific documentation on callbacks, check the API reference.

Distributed Training
=====================
We also provide a simple way to use PyTorch's :code:`DistributedDataParallel`
functionality to hold data across multiple GPUs. We use PyTorch's NCCL backend,
so the dependencies are totally shared with those of :code:`neuraloperator`. 

We currently have support for a pure-Python implementation of single-node, multi-GPU
training. In order to run training across multiple GPUs in our example script 
:code:`scripts/train_darcy.py`, simply adjust the configuration file to set 
:code:`config.distributed.use_distributed` to :code:`True`. 

You may need to adjust the batch size and world size in 
accordance with your specific use case. 


 


