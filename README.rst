.. image:: https://img.shields.io/pypi/v/neuraloperator
   :target: https://pypi.org/project/neuraloperator/
   :alt: PyPI

.. image:: https://github.com/NeuralOperator/neuraloperator/actions/workflows/test.yml/badge.svg
   :target: https://github.com/NeuralOperator/neuraloperator/actions/workflows/test.yml


#######################################################################
NeuralOperator: Learning in Infinite Dimensions
#######################################################################

``neuraloperator`` is a comprehensive library for 
learning neural operators in PyTorch.
It is the official implementation for Fourier Neural Operators 
and Tensorized Neural Operators.

Unlike regular neural networks, neural operators
enable learning mapping between function spaces, and this library
provides all of the tools to do so on your own data.

Neural operators are also resolution invariant, 
so your trained operator can be applied on data of any resolution.

Checkout the `documentation <https://neuraloperator.github.io/dev/index.html>`_ for more!

============
Installation
============

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


==========
Quickstart
==========

After you've installed the library, you can start training operators seamlessly:


.. code-block:: python

   from neuralop.models import FNO

   operator = FNO(n_modes=(32, 32), 
                  hidden_channels=64,
                  in_channels=2, 
                  out_channels=1)

Tensorization is also provided out of the box: you can improve the previous models
by simply using a Tucker Tensorized FNO with just a few parameters:

.. code-block:: python

   from neuralop.models import TFNO

   operator = TFNO(n_modes=(32, 32), 
                   hidden_channels=64,
                   in_channels=2, 
                   out_channels=1,
                   factorization='tucker',
                   implementation='factorized',
                   rank=0.05)

This will use a Tucker factorization of the weights. The forward pass
will be efficient by contracting directly the inputs with the factors
of the decomposition. The Fourier layers will have 5% of the parameters
of an equivalent, dense Fourier Neural Operator!

Checkout the `documentation <https://neuraloperator.github.io/dev/index.html>`_ for more!


Using with Weights and Biases
-----------------------------

Our ``Trainer`` natively supports logging to W&B. To use these features, create a file in
``neuraloperator/config`` called ``wandb_api_key.txt`` and paste your W&B API key there.
You can configure the project you want to use and your username in the main yaml configuration files.


============
Contributing
============

NeuralOperator is 100% open-source, and we welcome contributions from the community! 

Our mission for NeuralOperator is to provide access to well-documented, robust implementations of 
neural operator methods from foundations to the cutting edge. The library is primarily intended for 
methods that directly relate to operator learning: new architectures, meta-algorithms, training methods 
and benchmark datasets. We are also interested in integrating interactive examples that showcase operator 
learning in action on small sample problems.

If your work provides one of the above, we would be thrilled to integrate it into the library. 
Otherwise, if your work simply relies on a version of the NeuralOperator codebase, we recommend 
publishing your code separately using a procedure outlined in our
`developer's guide <https://neuraloperator.github.io/dev/dev_guide/index.html>`_, under the section 
"Publishing code built on the library". 

If you spot a bug or a typo in the documentation, or have an idea for a feature you'd like to see,
please report it on our `issue tracker <https://github.com/neuraloperator/neuraloperator/issues>`_, 
or even better, open a `Pull Request <https://github.com/neuraloperator/neuraloperator/pulls>`_. 

For detailed development setup, testing, and contribution guidelines, please refer to our `Contributing Guide <CONTRIBUTING.md>`_.


===============
Code of Conduct
===============

All participants are expected to uphold the `Code of Conduct <https://github.com/neuraloperator/neuraloperator/blob/main/CODE_OF_CONDUCT.md>`_ to ensure a friendly and welcoming environment for everyone.


=====================
Citing NeuralOperator
=====================

If you use NeuralOperator in an academic paper, please cite [1]_ ::

   @article{kossaifi2025librarylearningneuraloperators,
      author    = {Jean Kossaifi and
                     Nikola Kovachki and
                     Zongyi Li and
                     David Pitt and
                     Miguel Liu-Schiaffini and
                     Valentin Duruisseaux and
                     Robert Joseph George and
                     Boris Bonev and
                     Kamyar Azizzadenesheli and
                     Julius Berner and
                     Anima Anandkumar},
      title     = {A Library for Learning Neural Operators},
      journal   = {arXiv preprint arXiv:2412.10354},
      year      = {2025},
   }

and consider citing [2]_, [3]_::

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

   @article{berner2025principled,
      author    = {Julius Berner and
                     Miguel Liu-Schiaffini and
                     Jean Kossaifi and
                     Valentin Duruisseaux and
                     Boris Bonev and
                     Kamyar Azizzadenesheli and
                     Anima Anandkumar},
      title     = {Principled Approaches for Extending Neural Architectures to Function Spaces for Operator Learning},
      journal   = {arXiv preprint arXiv:2506.10973},
      year      = {2025},
   }


.. [1] Kossaifi, J., Kovachki, N., Li, Z., Pitt, D., Liu-Schiaffini, M., Duruisseaux, V., George, R., Bonev, B., Azizzadenesheli, K., Berner, J., and Anandkumar, A., "A Library for Learning Neural Operators", ArXiV, 2025. doi:10.48550/arXiv.2412.10354.

.. [2] Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., and Anandkumar A., "Neural Operator: Learning Maps Between Function Spaces", JMLR, 2021. doi:10.48550/arXiv.2108.08481.

.. [3] Berner, J., Liu-Schiaffini, M., Kossaifi, J., Duruisseaux, V., Bonev, B., Azizzadenesheli, K., and Anandkumar, A., "Principled Approaches for Extending Neural Architectures to Function Spaces for Operator Learning", arXiv preprint arXiv:2506.10973, 2025. https://arxiv.org/abs/2506.10973.
