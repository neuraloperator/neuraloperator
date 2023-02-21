.. image:: https://img.shields.io/pypi/v/neuraloperator
   :target: https://pypi.org/project/neuraloperator/
   :alt: PyPI

.. image:: https://github.com/NeuralOperator/neuraloperator/actions/workflows/test.yml/badge.svg
   :target: https://github.com/NeuralOperator/neuraloperator/actions/workflows/test.yml


===============
Neural Operator
===============

``neuraloperator`` is a comprehensive library for 
learning neural operators in PyTorch.
It is the official implementation for Fourier Neural Operators 
and Tensorized Neural Operators.

Unlike regular neural networks, neural operators
enable learning mapping between function spaces, and this library
provides all of the tools to do so on your own data.

NeuralOperators are also resolution invariant, 
so your trained operator can be applied on data of any resolution.


Installation
------------

Just clone the repository and install locally (in editable mode so changes in the code are immediately reflected without having to reinstall):

.. code::

  git clone https://github.com/NeuralOperator/neuraloperator
  cd neuraloperator
  pip install -e .
  
You can also just pip install the library:


.. code::
  
  pip install neuraloperator

Quickstart
----------

After you've installed the library, you can start training operators seemlessly:


.. code-block:: python

   from neuralop.models import TFNO

   operator = TFNO(n_modes=(16, 16), hidden_channels=64,
                   in_channels=3, out_channels=1)

Tensorization is also provided out of the box: you can improve the previous models
by simply using a Tucker Tensorized FNO with just a few parameters:

.. code-block:: python

   from neuralop.models import TFNO

   operator = TFNO(n_modes=(16, 16), hidden_channels=64,
                   in_channels=3, 
                   out_channels=1,
                   factorization='tucker',
                   implementation='factorized'
                   rank=0.05)

This will use a Tucker factorization of the weights. The forward pass
will be efficient by contracting directly the inputs with the factors
of the decomposition. The Fourier layers will have 5% of the parameters
of an equivalent, dense Fourier Neural Operator!

Checkout the `documentation <https://neuraloperator.github.io/neuraloperator/dev/index.html>`_ for more!

Using with weights and biases
-----------------------------

Create a file in `neuraloperator/config` called `wandb_api_key.txt` and paste your Weights and Biases API key there.
You can configure the project you want to use and your username in the main yaml configuration files.



