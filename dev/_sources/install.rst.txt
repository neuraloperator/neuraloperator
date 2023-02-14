=========================
Installing NeuralOperator
=========================

The package is called ``neuraloperator`` and provides all the tools for operator learning.
Once install, you can import it as ``neuralop``::

    import neuralop


Pre-requisite
=============

You will need to have Python 3 installed, as well as NumPy, Scipy, PyTorch, TensorLy and TensorLy-Torch.
If you are starting with Python or generally want a pain-free experience, 
I recommend you install the `Anaconda distribiution <https://www.anaconda.com/download/>`_. It comes with all you need shipped-in and ready to use!


Installing with pip (recommended)
=================================


Simply run, in your terminal::

   pip install -U neuraloperator

(the `-U` is optional, use it if you want to update the package).


Cloning the github repository
=============================

Clone the repository and cd there::

   git clone https://github.com/NeuralOperator/neuraloperator
   cd torch


You can install the requirements easily::

   pip install -r requirements.txt


Then install the package (here in editable mode with `-e` or equivalently `--editable`::

   pip install -e .


Running the tests
=================

Uni-testing is an important part of this package.
You can run all the tests using `pytest`::

   pip install pytest
   pytest neuralop

Building the documentation
==========================

You will need to install the dependencies::

   cd doc
   pip install -r requirements_doc.txt


You are now ready to build the doc (here in html)::

   make html

The results will be in ``build/html`` (the main page will be ``build/html/index.html``)
