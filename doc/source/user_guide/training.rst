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
Using the small Darcy flow dataset we provide along with the library, a :code:`Trainer`
is implemented as follows:

.. code:: python

    from neuralop.models import FNO2d
    from neuralop.datasets import load_darcy_flow_small
    from neuralop.training import Trainer, LpLoss

    train_loader, test_loaders, output_encoder = load_darcy_flow_small(
        n_train=100, batch_size=4, 
        test_resolutions=[16, 32], n_tests=[50, 50], test_batch_sizes=[4, 2],
        )
    
    model = FNO2d(n_modes=16,
                  hidden_channels=32)

    trainer = Trainer(
        
    )


We were inspired by PyTorch-Lightning_ to create a 
.. _Pytorch-Lightning: https://lightning.ai/
