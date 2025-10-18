.. _dev_guide:

=================
Development guide
=================

This guide provides essential information for developers contributing to NeuralOperator.

We welcome new contributions to the library! Our mission for NeuralOperator is to 
provide access to well-documented, robust implementations of neural operator methods 
from foundations to the cutting edge. 
The library is primarily intended for methods that directly relate to operator learning:
new architectures, meta-algorithms, training methods and benchmark datasets. 
We are also interested in integrating interactive examples that showcase operator 
learning in action on small sample problems.

If your work provides one of the above, we would be thrilled to integrate it into the 
library. Otherwise, if your work simply relies on a version of the NeuralOperator 
codebase, we recommend publishing your code separately using a procedure outlined 
in the :ref:`Publishing code built on the library <publishing_code_built_on_neuralop>` 
section.

.. raw:: html

   <div style="margin-top: 4em;"></div>

Development Setup
------------------

.. _fork-and-clone:

Fork and Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Fork the repository** on GitHub by clicking the "Fork" button

2. **Clone the library and connect your fork**:

   .. code-block:: bash

      git clone https://github.com/neuraloperator/neuraloperator.git
      cd neuraloperator
      git remote rename origin upstream
      git remote add origin https://github.com/<YOUR_GIT_NAME>/neuraloperator.git
      git remote -v

   This should show:

   .. code-block:: text

      origin  https://github.com/<YOUR_GIT_NAME>/neuraloperator.git (fetch)
      origin  https://github.com/<YOUR_GIT_NAME>/neuraloperator.git (push)
      upstream        https://github.com/neuraloperator/neuraloperator.git (fetch)
      upstream        https://github.com/neuraloperator/neuraloperator.git (push)

.. raw:: html

   <div style="margin-top: 2em;"></div>

Set Up Environment
~~~~~~~~~~~~~~~~~~

1. **Create a virtual environment**:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

2. **Install development dependencies**:

   .. code-block:: bash

      pip install -e .[dev]
      # OR
      pip install neuraloperator[dev]

.. raw:: html

   <div style="margin-top: 4em;"></div>


Development Workflow
---------------------

.. _create-branch:

.. raw:: html

   <div style="margin-top: 2em;"></div>

Create a Branch and Make Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create and switch to a new branch
   git checkout -b feature/your-feature-name

   # Make your changes
   # ... edit files ...

   # Stage and commit your changes
   git add .
   git commit -m "Add: brief description of your changes

   - More detailed explanation of what you changed
   - Why you made these changes
   - Any relevant context"

   # Push to your fork
   git push -u origin feature/your-feature-name

.. raw:: html

   <div style="margin-top: 2em;"></div>

Write Tests
~~~~~~~~~~~

**Always add tests for new functionality!**

- **For models**: Add tests in ``neuralop/models/tests/test_your_model.py``
- **For layers**: Add tests in ``neuralop/layers/tests/test_your_layer.py``
- **For utilities**: Add tests in the appropriate test directory

**Example test structure:**

.. code-block:: python

   import pytest
   import torch
   from neuralop.models import YourModel

   class TestYourModel:
       def test_forward_pass(self):
           model = YourModel(...)
           x = torch.randn(1, 3, 32, 32)
           output = model(x)
           assert output.shape == expected_shape
       
       def test_gradient_flow(self):
           # Test that gradients flow properly
           pass

.. raw:: html

   <div style="margin-top: 2em;"></div>

Run Tests and Quality Checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest neuralop

   # Run tests with verbose output
   pytest neuralop -v

   # Run specific test file
   pytest neuralop/models/tests/test_your_model.py

   # Format code
   black .

.. raw:: html

   <div style="margin-top: 2em;"></div>

Submit a Pull Request
~~~~~~~~~~~~~~~~~~~~~

After having forked and cloned the repository as described in 
the :ref:`Fork and Clone the Repository <fork-and-clone>` section, 
and created a branch and committed changes as described in 
the :ref:`Create a Branch and Make Changes <create-branch>` section, 
you can now submit your pull request:

1. **Push your branch to your fork**:

   .. code-block:: bash

      git push -u origin feature/your-feature-name


2. **Go to your fork on GitHub** and you should see a banner suggesting to  
   create a pull request, or click "Compare & pull request".

3. **Fill out the PR description**:

   - Provide a clear title that describes your changes
   - Write a detailed description explaining what you've changed or added
   - Reference any related issues using `#issue_number`
   - Include screenshots or examples if applicable

4. **Submit the pull request** by clicking "Create pull request".

Ensure your PR description clearly communicates what you have changed or added, why you made these changes, and any relevant context for reviewers.

.. raw:: html

   <div style="margin-top: 4em;"></div>

Development Guidelines
----------------------

.. raw:: html

   <div style="margin-top: 2em;"></div>

Code Style
~~~~~~~~~~~

- **Follow PEP 8** style guidelines for Python code
- **Use meaningful names** for variables, functions, and classes
- **Write clear docstrings** using NumPy docstring format
- **Add type hints** where appropriate

.. raw:: html

   <div style="margin-top: 2em;"></div>

Code Formatting
~~~~~~~~~~~~~~~

Before submitting, ensure your code follows our style guide:

.. code-block:: bash

   # Format with black
   black .

Validate every update made by the ``black`` command

.. raw:: html

   <div style="margin-top: 2em;"></div>

Testing Requirements
~~~~~~~~~~~~~~~~~~~~

- **All new code must have tests**
- **Run the full test suite** before submitting PRs
- **Use descriptive test names** that explain what is being tested
- **Test both normal operation and edge cases**
- **Aim for complete code coverage** for new functionality

.. raw:: html

   <div style="margin-top: 2em;"></div>

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~~

The HTML documentation is built using Sphinx:

.. code-block:: bash

   cd doc
   make html

This builds the docs in ``./doc/build/html``. Note that documentation requires additional dependencies from ``./doc/requirements_doc.txt``.

To view documentation locally:

.. code-block:: bash

   cd doc/build/html
   python -m http.server 8000
   # Visit http://localhost:8000

.. raw:: html

   <div style="margin-top: 2em;"></div>

Git Best Practices
~~~~~~~~~~~~~~~~~~

- **Write clear, descriptive commit messages**
- **Keep commits focused and atomic** (one logical change per commit)
- **Rebase your branch** on main before submitting PRs
- **Use conventional commit format** when possible (feat:, fix:, docs:, etc.)

.. raw:: html

   <div style="margin-top: 2em;"></div>

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~~

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

.. raw:: html

   <div style="margin-top: 4em;"></div>

Adding a New Model or Layer
----------------------------

We welcome various types of contributions:

- **Bug fixes** - Report and fix bugs
- **New features** - Add new models, layers, or datasets
- **Examples** - Create new examples or improve existing ones
- **Performance** - Optimize existing code

.. raw:: html

   <div style="margin-top: 2em;"></div>

Adding a New Neural Operator Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add a new neural operator model:

1. **Create a new file** in ``neuralop/models/your_model.py``
2. **Implement the model** as a subclass of ``BaseModel``
3. **Add comprehensive tests** in ``neuralop/models/tests/test_your_model.py``
4. **Update imports** in the appropriate ``__init__.py`` files
5. **Add documentation** with examples and mathematical formulations

**Example structure:**

.. code-block:: python

   from neuralop.models import BaseModel

   class YourModel(BaseModel):
       """Your model docstring here."""
       
       def __init__(self, ...):
           super().__init__()
           # Your implementation
       
       def forward(self, x):
           # Your forward pass
           return x

.. raw:: html

   <div style="margin-top: 2em;"></div>

Adding a New Layer
~~~~~~~~~~~~~~~~~~~

To add a new layer:

1. **Create a new file** in ``neuralop/layers/your_layer.py``
2. **Ensure the layer** is a subclass of ``torch.nn.Module``
3. **Add comprehensive tests** in ``neuralop/layers/tests/test_your_layer.py``
4. **Update imports** in the appropriate ``__init__.py`` files

.. note::

   **💡 Pro Tip**: For bonus points, add an interactive example featuring your new method to ``./examples``. This helps both us and you: the simpler it is for new users to understand and adapt your method, the more visibility it will get!

.. raw:: html

   <div style="margin-top: 4em;"></div>

Getting Help
------------

If you need assistance while contributing to NeuralOperator, here are the best ways to get help:

.. raw:: html

   <div style="margin-top: 2em;"></div>

Before Asking for Help
~~~~~~~~~~~~~~~~~~~~~~

1. **Check the documentation** and existing issues
2. **Search GitHub issues** for similar problems
3. **Provide a minimal reproducible example**
4. **Include error messages** and stack traces
5. **Specify your environment** (OS, Python version, PyTorch version)

.. raw:: html

   <div style="margin-top: 2em;"></div>

Contact Methods
~~~~~~~~~~~~~~~

- **GitHub Issues**: Create an issue for bugs or feature requests
- **GitHub Discussions**: Use for questions and ideas
- **Documentation**: Check the `official documentation <https://neuraloperator.github.io/>`_

.. raw:: html

   <div style="margin-top: 4em;"></div>


License
-------

By contributing to NeuralOperator, you agree that your contributions will be licensed as described in the `LICENSE` file in the root directory of this source tree.

.. raw:: html

   <div style="margin-top: 4em;"></div>

Acknowledgments
---------------

Thank you for contributing to NeuralOperator! Your contributions help make this library better for the entire scientific machine learning community.

