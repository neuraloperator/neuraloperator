.. _dev_guide:

=================================
NeuralOperator Developer's Guide
=================================

This guide provides essential information for developers contributing to NeuralOperator.

~~~~~~~~

Installation and Setup
-----------------------

To get started with development, fork the repository on GitHub, then download your fork:

.. code-block:: bash

    git clone https://github.com/YOURNAME/neuraloperator.git
    cd neuraloperator
    pip install -e .[dev]

~~~~~~~~

Code Style and Standards
------------------------

To ensure code clarity and future maintainability, NeuralOperator adheres to a few style principles:

Commenting and docstrings
+++++++++++++++++++++++++

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

~~~~~~~~

Contributing
------------

Follow these steps when making contributions:

1. Create a new branch for your feature or bug fix:

   .. code-block:: bash
      
       git checkout -b feature-branch

2. Write clean, well-documented code. 

3. Add or update tests in appropriate directory. For instance, if your feature adds a model
in ``neuralop/models/mymodel.py``, you would add tests to `neuralop/models/tests/test_mymodel.py`

4. Run the test suite:
.. code-block:: bash
    
    pytest neuralop

5. Submit a pull request (PR) on GitHub from your branch to the upstream origin/main. 
Ensure your PR clearly communicates what you've changed or added. 


Extending the Library
+++++++++++++++++++++

To add a new neural operator model:

1. Create a new file in ``neuralop/models/``.
2. Implement the model as a subclass of ``BaseModel``.

To add a layer:

1. Create a new file in ``neuralop/layers``
2. Ensure the layer is a subclass of ``torch.nn.Module``.

.. note ::
    For optional bonus points, add an interactive example featuring your new method to ``./examples``.
    This helps both us and you: the simpler it is for new users to understand and adapt your method, 
    the more visibility it will get! 

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
