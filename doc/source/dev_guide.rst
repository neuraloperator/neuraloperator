.. _dev_guide:

=================================
NeuralOperator Developer's Guide
=================================

This guide provides essential information for developers contributing to NeuralOperator.

~~~~~~~~

Installation and Setup
-----------------------

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

**For CUDA installations:**

.. code-block:: bash

    pip install torch torchvision --index-url https://download.pytorch.org/whl/<YOUR_CUDA_VERSION>

**For CPU installations:**

.. code-block:: bash

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Next, install the library locally in editable mode, along with the ``".[dev]"`` tag to install all development dependencies 

.. code-block:: bash
    
    pip install -e .[dev]

To test your installation, run python in interactive mode and import the library as ``neuralop`` to ensure it is properly built:

.. code-block:: bash

    $ python
    Python 3.10.14 (main, Month Day Year, Time of Day) [GCC VERSION] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import neuralop
    >>> 

You have now successfully built ``neuralop``. For instructions on contributing code, see below. 

~~~~~~~~

Contributing code
-----------------

We welcome new contributions to the library! Our mission for NeuralOperator is to provide access
to well-documented, robust implementations of neural operator methods from foundations to the cutting edge. 
The library is primarily intended for methods that directly relate to operator learning: new architectures, 
meta-algorithms, training methods and benchmark datasets. We are also interested in integrating interactive
examples that showcase operator learning in action on small sample problems. 

If your work provides one of the above, we would be thrilled to integrate it into the library. Otherwise, if your
work simply relies on a version of the NeuralOperator codebase, we recommend publishing your code separately using 
a procedure outlined :ref:`here <publishing_code_built_on_neuralop>`.

Extending NeuralOperator
++++++++++++++++++++++++

To add a new neural operator model:

1. Create a new file in ``neuralop/models/``.
2. Implement the model as a subclass of ``BaseModel``.
3. Add a parametrized unit test file in ``neuralop/models/tests``.

To add a layer:

1. Create a new file in ``neuralop/layers``
2. Ensure the layer is a subclass of ``torch.nn.Module``.
3. Add a parametrized unit test file in ``neuralop/layers/tests``.

.. note ::
    For optional bonus points, add an interactive example featuring your new method to ``./examples``.
    This helps both us and you: the simpler it is for new users to understand and adapt your method, 
    the more visibility it will get! 

Code Style and Standards
++++++++++++++++++++++++

To ensure code clarity and future maintainability, NeuralOperator adheres to simple style principles.

In general, docstrings use the NumPy format:

.. code-block:: python

    def function(arg1: type1, arg2: type2=default)
        """
        Parameters
        ----------
        arg1 : type1
            description of what arg1 'means'
            for the function's behavior
        arg2 : type2, optional
            description arg2
            by default default
        """

For *classes*, this docstring should go directly below the class declaration:

.. code-block:: python

    class MyClass(Superclass):
        """
        docstring goes here
        """
        def __init__(self, ...):
        # Full NumPy docstring not needed here.
        
We also adhere to good in-line commenting practices. When a block's function is not
obvious on its own, add in-line comments with a brief description. For tensor operations, 
shape annotations are especially helpful where applicable.

Submitting Contributions
++++++++++++++++++++++++

Follow these steps when making contributions:

1. Create a new branch for your feature or bug fix:

   .. code-block:: bash
      
       git checkout -b feature-branch

2. Write clean, well-documented code (see above).

3. Add or update tests in appropriate directory. For instance, if your feature adds a model
in ``neuralop/models/mymodel.py``, you would add tests to ``neuralop/models/tests/test_mymodel.py``

4. Run the test suite:

.. code-block:: bash
    
    pytest neuralop

5. Submit a pull request (PR) on GitHub from your branch to the upstream origin/main. 
Ensure your PR description clearly communicates what you've changed or added. 

.. _publishing_code_built_on_neuralop:

Publishing code built on the library
++++++++++++++++++++++++++++++++++++

If you plan to use ``neuralop`` as the base of a project, we suggest the following workflow:

1. First, set up a clean virtual environment.

2. Then install ``neuralop`` via ``pip``. There are two ways to do this:

* To install the latest PyPI release of the library, simply run: 

.. code-block:: bash

    pip install neuralop

* If you need access to functionality that was added after the last PyPI release, you can pip install the library from a git commit hash:

Go to the repository's `commit history page <https://github.com/neuraloperator/neuraloperator/commits/main/>`_ and locate the commit
hash that corresponds to the repository state at which you want to install the repo. For most use cases, this will be the most recent commit. 

To find the commit hash, click the commit title, which will take you to the commit's url. The hash will be the last component of the commit's URL,
e.g. ``https://github.com/neuraloperator/neuraloperator/commit/<COMMIT_HASH>``. Copy this hash to your clipboard.

Then, use ``pip`` to install the library with the hash you just saved. 

.. code-block:: bash
    
    pip install git+https://github.com/neuraloperator/neuraloperator.git@<COMMIT_HASH>

Once installed, if you plan to implement new functionality, like a new model or dataset, we recommend you **subclass** the functionality
you need. For instance, to create a modified ``FNO`` that performs extra steps during the forward pass:

.. code-block:: python

    from neuralop.models import FNO

    # other imports here

    class MyFNO(FNO):
        def __init__(self, ...)
            super().__init__()
        
        def forward(self, x, ...)
            # do your special operations here
            x = my_operations(x, ...)
            # pass through the standard FNO.forward()
            x = super().forward(x, ...)

            # more operations could go here
            x = my_other_operations(x, ...)

            return x

~~~~~~~~

Debugging and Troubleshooting
-----------------------------

- Use `torch.set_detect_anomaly(True)` for debugging NaN gradients.
- Check GPU memory usage with `nvidia-smi`.
- Ensure dependencies are up to date with `pip list --outdated`.

~~~~~~~~

Contact
-------

For questions or issues, create a GitHub issue or reach out on the discussion forum.
