=========================
Installing NeuralOperator
=========================

The ``neuraloperator`` Python package provides all necessary tools for implementing operator learning.
Once installed, you can import it as ``neuralop``::

    import neuralop

~~~~~~~~~~~~~~~~~~~~~~~~~~

Basic Install
--------------

~~~~~~~~~~~~~~~~~~~~~~~~~~
Pre-requisites
++++++++++++++

You will need to have Python 3 installed, as well as NumPy, Scipy, PyTorch, TensorLy and TensorLy-Torch.
If you are starting with Python or generally want a pain-free experience, we recommend that you 
install the `Anaconda distribution <https://www.anaconda.com/download/>`_. It comes ready to use with all prerequisite packages.

~~~~~~~~~~~~~~~~~~~~~~~~~~

Installing with pip
+++++++++++++++++++

We periodically package `neuraloperator` for release on PyPI. This version is not guaranteed to be up-to-date with
the latest changes to our code. 

To install via pip, simply run, in your terminal::

   pip install -U neuraloperator

(the `-U` is optional, use it if you want to update the package).

~~~~~~~~~~~~~~~~~~~~~~~~~~

Building ``neuraloperator`` from source
+++++++++++++++++++++++++++++++++++++++

First ensure that you are in an environment with Python 3, NumPy, SciPy, PyTorch, TensorLy and TensorLy-Torch. 
Then clone the repository and cd there::

   git clone https://github.com/neuraloperator/neuraloperator
   cd neuraloperator


Then, install the requirements ::

   pip install -r requirements.txt


Then install the package (here in editable mode with `-e`, or equivalently `--editable`)::

   pip install -e .

~~~~~~~~~~~~~~~~~~~~

Advanced dependencies
---------------------

.. _open3d_dependency :
Fast 3D spatial computing with Open3D
+++++++++++++++++++++++++++++++++++++

To accelerate spatial computing for 3D applications, we include 
`Open3D <https://github.com/isl-org/Open3D>`_ as an optional dependency. Open3D includes
utilities for reading 3D mesh files and fast 3D neighbor search. To install::

   pip install open3d

Note that Open3D is only
compatible with specific builds of PyTorch and CUDA. Check the sub-package 
`Open3D-ML <https://github.com/isl-org/Open3D-ML>`_ documentation for more details. 

~~~~~~~~~~~~~~~~~~~~~

.. _torch_scatter_dependency :

Sparse computations with PyTorch-Scatter
++++++++++++++++++++++++++++++++++++++++

We use the package ``torch_scatter`` (available on PyPI) to speed up neighborhood reductions during 
the final stage of the ``IntegralTransform`` layer, linked here: :ref:`gno_api`.

``torch_scatter`` uses compiled CUDA kernels 
to perform this reduction, providing both memory efficiency and speed advantages over a native PyTorch implementation.

However, since these kernels are precompiled, it is crucial to install the version of ``torch_scatter`` that is built for
your specific combination of OS, PyTorch and CUDA. All available wheels can be found `here <https://data.pyg.org/whl/>`_.  

.. note :: 
    The latest PyTorch version for which ``torch_scatter`` is built is 2.5.1, so you should downgrade if you are using 
    the latest (``torch>=2.6.0``) release:


To force-downgrade PyTorch, run 

.. code ::

   pip install -U torch==2.5.1

~~~~~~~~~~~~~~~~~~~~~~

Developer Setup
---------------

Install the library for development:
++++++++++++++++++++++++++++++++++++

To get started with development, fork the repository on GitHub, then download your fork. After cloning, 
move to the top level of the repo in your terminal. 

.. code-block:: bash

    git clone https://github.com/YOURNAME/neuraloperator.git
    cd neuraloperator

.. note:: 

    To manage the library's dependencies and minimize the risk of conflicts with other packages installed in your
    default local Python environment, we recommend developing and building ``neuraloperator`` in a fresh Python
    virtual environment. 

.. warning::

    Previous versions of our installation and build guides recommended starting with the Anaconda distribution. Our library
    depends on `PyTorch <https://pytorch.org>`_. As of spring 2025, PyTorch has stopped releasing updates to the Anaconda
    distribution. For the latest PyTorch we recommend you use ``pip``. 

Create your virtual environment and store it in the top level:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate

Install PyTorch. These are generic instructions; if you require a specific build, or a specific CUDA version, your installation
command will vary. Check PyTorch's `getting started page <https://pytorch.org/get-started/locally/>`_ for more detailed instructions


To test your installation, run python in interactive mode and import the library as ``neuralop`` to ensure it is properly built:

.. code-block:: bash

    $ python
    Python 3.10.14 (main, Month Day Year, Time of Day) [GCC VERSION] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import neuralop
    >>> 

You have now successfully built ``neuralop``. For instructions on contributing code, see below. 


Running the tests
+++++++++++++++++

Unit-testing is an important part of this package.
You can run all the tests using `pytest`::

   pip install -r requirements_dev.txt
   pytest neuralop

Building the documentation
++++++++++++++++++++++++++

You will need to install the dependencies::

   cd doc
   pip install -r requirements_doc.txt


You are now ready to build the doc (here in html)::

   make html

The results will be in ``build/html`` (the main page will be ``build/html/index.html``)
