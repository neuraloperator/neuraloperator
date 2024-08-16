.. _getting_started :

Getting Started
===========

*NeuralOperator* aims to provide you with all the tools 
to easily use, build and train neural operators for your own applications
and learn mapping between function spaces, in PyTorch.

Intro to operator learning
----------------------------
To get a better feel for the theory behind our neural operator models, see :ref:`neuralop_intro` and :ref:`fno_intro`.

~~~~~~~~~~~~

Interactive examples with code
----------------------------
To get up to speed on the code, and look through some interactive examples to help you hit the ground running,
check out :ref:`gallery_examples`.

~~~~~~~~~~~~

NeuralOperator library structure
---------------------------------

Here are the main components of the library:

================================= ================================
Module                             Description
================================= ================================
:mod:`neuralop`                   Main library 
:mod:`neuralop.models`            Full ready-to-use neural operators
:mod:`neuralop.layers`            Individual layers to build neural operators
:mod:`neuralop.datasets`          Convenience PyTorch data loaders for PDE datasets
:mod:`neuralop.training`          Utilities to train neural operators end-to-end
================================= ================================

The full API documentation is provided in :ref:`api_ref`.

Finally, if you're building the library from source, your repository will also include the following directories:

================================= ================================
Directory                         Description
================================= ================================
:mod:`scripts`                    Training recipe scripts for our models on sample problems
:mod:`examples`                   More documented interactive examples, seen in 
================================= ================================
