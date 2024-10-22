.. image:: https://img.shields.io/pypi/v/neuraloperator
   :target: https://pypi.org/project/neuraloperator/
   :alt: PyPI

.. image:: https://github.com/NeuralOperator/neuraloperator/actions/workflows/test.yml/badge.svg
   :target: https://github.com/NeuralOperator/neuraloperator/actions/workflows/test.yml


===============
NeuralOperator: Learning in Infinite Dimensions
===============

``neuraloperator`` is a comprehensive library for 
learning neural operators in PyTorch.
It is the official implementation for Fourier Neural Operators 
and Tensorized Neural Operators.

Unlike regular neural networks, neural operators
enable learning mapping between function spaces, and this library
provides all of the tools to do so on your own data.

Neural operators are also resolution invariant, 
so your trained operator can be applied on data of any resolution.


Installation
------------

Just clone the repository and install locally (in editable mode so changes in the code are 
immediately reflected without having to reinstall):

.. code::

  git clone https://github.com/NeuralOperator/neuraloperator
  cd neuraloperator
  pip install -e .
  pip install -r requirements.txt

You can also just pip install the most recent stable release of the library 
on `PyPI <https://pypi.org/project/neuraloperator/>`_:


.. code::
  
  pip install neuraloperator

Quickstart
----------

After you've installed the library, you can start training operators seamlessly:


.. code-block:: python

   from neuralop.models import FNO

   operator = FNO(n_modes=(16, 16), hidden_channels=64,
                   in_channels=3, out_channels=1)

Tensorization is also provided out of the box: you can improve the previous models
by simply using a Tucker Tensorized FNO with just a few parameters:

.. code-block:: python

   from neuralop.models import TFNO

   operator = TFNO(n_modes=(16, 16), hidden_channels=64,
                   in_channels=3, 
                   out_channels=1,
                   factorization='tucker',
                   implementation='factorized',
                   rank=0.05)

This will use a Tucker factorization of the weights. The forward pass
will be efficient by contracting directly the inputs with the factors
of the decomposition. The Fourier layers will have 5% of the parameters
of an equivalent, dense Fourier Neural Operator!

Checkout the `documentation <https://neuraloperator.github.io/dev/index.html>`_ for more!

Using with weights and biases
-----------------------------

Create a file in ``neuraloperator/config`` called ``wandb_api_key.txt`` and paste your Weights and Biases API key there.
You can configure the project you want to use and your username in the main yaml configuration files.

===============
Contributing
===============

NeuralOperator is 100% open-source, and we welcome all contributions from the community! 
If you spot a bug or a typo in the documentation, or have an idea for a feature you'd like to see,
please report it on our `issue tracker <https://github.com/neuraloperator/neuraloperator/issues>`_, 
or even better, open a Pull-Request on `GitHub <https://github.com/neuraloperator/neuraloperator>`_. 

NeuralOperator has additional dependencies for development, which can be found in ``requirements_dev.txt``:

.. code::
   
   pip install -r requirements_dev.txt

Code formatting
----------------

Before you submit your changes, you should make sure your code adheres to our style-guide. The
easiest way to do this is with ``black``:

.. code::

   black .

Running the tests
------------------

Testing and documentation are an essential part of this package and all
functions come with unit-tests and documentation. The tests are run using the
pytest package. 
    
To run the tests, simply run, in the terminal:

.. code::

    pytest -v neuralop

Building documentation
-----------------------
The HTML for our documentation website is built using ``sphinx``. The documentation
is built from inside the ``doc`` folder. 

.. code::

   cd doc
   make html

This will build the docs in ``./doc/build/html``.

Note that the documentation requires other dependencies installable from ``./doc/requirements_doc.txt``. 

To view the documentation locally, run:

.. code::

   cd doc/build/html
   python -m http.server [PORT_NUM]

The docs will then be viewable at ``localhost:PORT_NUM``.

    
Citing
------

If you use NeuralOperator in an academic paper, please cite [1]_, [2]_::

   @misc{li2020fourier,
      title={Fourier Neural Operator for Parametric Partial Differential Equations}, 
      author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
      year={2020},
      eprint={2010.08895},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
   }

   @article{kovachki2021neural,
      author    = {Nikola B. Kovachki and
                     Zongyi Li and
                     Burigede Liu and
                     Kamyar Azizzadenesheli and
                     Kaushik Bhattacharya and
                     Andrew M. Stuart and
                     Anima Anandkumar},
      title     = {Neural Operator: Learning Maps Between Function Spaces},
      journal   = {CoRR},
      volume    = {abs/2108.08481},
      year      = {2021},
   }


.. [1] Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar A., “Fourier Neural Operator for Parametric Partial Differential Equations”, ICLR, 2021. doi:10.48550/arXiv.2010.08895.

.. [2] Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., and Anandkumar A., “Neural Operator: Learning Maps Between Function Spaces”, JMLR, 2021. doi:10.48550/arXiv.2108.08481.
