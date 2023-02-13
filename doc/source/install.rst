==========================
Installing neural-operator
==========================

The package is called ``neural-operator`` and provides all the tools for operator learning.
Once install, you can import it as ``neuralop``::

    import neuralop


Pre-requisite
=============

You will need to have Python 3 installed, as well as NumPy, Scipy and PyTorch.
If you are starting with Python or generally want a pain-free experience, I recommend you install the `Anaconda distribiution <https://www.anaconda.com/download/>`_. It comes with all you need shipped-in and ready to use!


Installing with pip (recommended)
=================================


Simply run, in your terminal::

   pip install -U neural-operator

(the `-U` is optional, use it if you want to update the package).


Cloning the github repository
=============================

Clone the repository and cd there::

   git clone https://github.com/neural-operators/neural-operator
   cd torch

Then install the package (here in editable mode with `-e` or equivalently `--editable`::

   pip install -e .

Running the tests
=================

Uni-testing is an vital part of this package.
You can run all the tests using `pytest`::

   pip install pytest
   pytest neuralop

Building the documentation
==========================

You will need to install slimit and minify::

   pip install slimit rcssmin

You are now ready to build the doc (here in html)::

   make html

The results will be in ``_build/html``

