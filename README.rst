.. image:: https://img.shields.io/pypi/v/neuraloperator
   :alt: PyPI

.. image:: https://github.com/NeuralOperator/neuraloperator/actions/workflows/test.yml/badge.svg
   :target: https://github.com/NeuralOperator/neuraloperator/actions/workflows/test.yml


===============
Neural Operator
===============

This repository provides all the tools to learn neural operators, i.e. mapping between function spaces, in PyTorch.

Installation
------------

Just clone the repository and install locally (in editable mode so changes in the code are immediately reflected without having to reinstall):

.. code::

  git clone https://github.com/NeuralOperator/neuraloperator
  cd neuraloperator
  pip install -e .


Using with weights and biases
-----------------------------

Create a file in `neuraloperator/config` called `wandb_api_key.txt` and paste your Weights and Biases API key there.
You can configure the project you want to use and your username in the main yaml configuration files.



