=========================
Installing NeuralOperator
=========================

The ``neuraloperator`` Python package provides all necessary tools for implementing operator learning.
Once installed, you can import it as ``neuralop``::

    import neuralop


Pre-requisites
===============

You will need to have Python 3 installed, as well as NumPy, Scipy, PyTorch, TensorLy and TensorLy-Torch.
If you are starting with Python or generally want a pain-free experience, we recommend that you 
install the `Anaconda distribiution <https://www.anaconda.com/download/>`_. It comes ready to use with all prerequisite packages.


Installing with pip
=================================

We periodically package `neuraloperator` for release on PyPI. This version is not guaranteed to be up-to-date with
the latest changes to our code. 

To install via pip, simply run, in your terminal::

   pip install -U neuraloperator

(the `-U` is optional, use it if you want to update the package).

Building ``neuraloperator`` from source
========================================

First clone the repository and cd there::

   git clone https://github.com/neuraloperator/neuraloperator
   cd neuraloperator


Then can install the requirements ::

   pip install -r requirements.txt


Then install the package (here in editable mode with `-e`, or equivalently `--editable`::

   pip install -e .


Running the tests
=================

Uni-testing is an important part of this package.
You can run all the tests using `pytest`::

   pip install -r requirements_dev.txt
   pytest neuralop

Building the documentation
==========================

You will need to install the dependencies::

   cd doc
   pip install -r requirements_doc.txt


You are now ready to build the doc (here in html)::

   make html

The results will be in ``build/html`` (the main page will be ``build/html/index.html``)
