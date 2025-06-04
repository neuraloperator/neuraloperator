Documentation
=============

Code Style and Standards
------------------------

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
