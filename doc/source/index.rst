:no-toc:
:no-localtoc:
:no-pagination:

.. neural-operator documentation

.. only:: html

   .. raw:: html

      <br/><br/>

.. only:: html

   .. raw:: html 
   
      <div class="has-text-centered">
         <h2> Neural Operators in PyTorch </h2>
      </div>
      <br/><br/>

.. only:: latex

   Neural Operators in PyTorch
   ===========================

.. image:: _static/logos/neuraloperator_logo_long.png
   :align: center
   :width: 500



NeuralOperator is a comprehensive PyTorch library for learning neural operators,
containing the official implementation of Fourier Neural Operators and other neural operator architectures.

NeuralOperator is part of the PyTorch Ecosystem, check the PyTorch `announcement <https://pytorch.org/blog/neuraloperatorjoins-the-pytorch-ecosystem>`_! 


Unlike regular neural networks, neural operators enable learning mapping between function spaces, 
and this library provides all of the tools to do so on your own data. Neural operators are 
resolution invariant, so your trained operator can be applied on data of any resolution.

.. raw:: html

   <br/>


Quickstart
==========

This guide will walk you through the standard ML workflow of loading data, creating a neural operator, 
training it on the data and saving the trained model for later use. (Check out :ref:`gallery_examples` for more info)

First install the library ``pip install neuraloperator`` (see :doc:`install` for more options).

To create a Fourier Neural Operator model:


.. code-block:: python

   from neuralop.models import FNO

   operator = FNO(
      n_modes=(16, 16), 
      hidden_channels=64,
      in_channels=2, 
      out_channels=1
   )

To save the weights of the trained model:

.. code-block:: python

   model.save_checkpoint(save_folder='./checkpoints/', save_name='example_fno')

And to load the weights later:

.. code-block:: python
   
   from neuralop.models import FNO
   model = FNO.from_checkpoint(save_folder='./checkpoints/', save_name='example_fno')

``neuraloperator`` comes prepackaged with an example dataset of flows governed by the Darcy flow equation. 

To import the data:

.. code-block:: python

   import torch
   from neuralop.data.datasets import load_darcy_flow_small

   train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, 
        batch_size=32, 
        n_tests=[100],
        test_resolutions=[32], 
        test_batch_sizes=[32],
   )

Similar to the API provided by ``torchvision``, this dataset includes training and test data for use in standard PyTorch training loops,
as well as a ``preprocessor`` object that automates the transforms to convert the data into the form best understood by the model. 

We provide a ``Trainer`` object that automates the logic of a basic neural operator training loop to speed up experimentation (see :doc: `auto_examples` for more information).

.. code-block:: python

   from neuralop.training import Trainer

   # Create the trainer
   trainer = Trainer(
      model=model, 
      n_epochs=20,
      data_processor=data_processor,
      verbose=True)

   # train the model
   trainer.train(
      train_loader=train_loader,
      test_loaders=test_loaders,
      optimizer=optimizer,
      scheduler=scheduler, 
      regularizer=False, 
      training_loss=train_loss,
      eval_losses=eval_losses
   )

Weight tensorization is also provided out of the box: you can improve the previous models
by simply using a Tucker Tensor FNO with fewer parameters:

.. code-block:: python

   from neuralop.models import TFNO

   operator = TFNO(
      n_modes=(16, 16), 
      hidden_channels=64,
      in_channels=2, 
      out_channels=1,
      factorization='tucker',
      implementation='factorized',
      rank=0.1
   )

This will use a Tucker factorization of the weights. The forward pass
will be efficient by contracting directly the inputs with the factors
of the decomposition. The Fourier layers will have 10% of the parameters
of an equivalent, dense Fourier Neural Operator!


.. toctree::
   :maxdepth: 2
   :hidden:

   install
   theory_guide/index
   user_guide/index
   modules/api
   auto_examples/index
   dev_guide/index

.. only:: html

   .. raw:: html

      <br/> 
      <br/>

      <div class="container has-text-centered">
      <a class="button is-medium is-dark is-primary" href="install.html">
         Install
      </a>
      </div>


