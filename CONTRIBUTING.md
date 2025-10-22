# Contributing to NeuralOperator

We welcome new contributions to the library! Our mission for NeuralOperator is to provide access to well-documented, robust implementations of neural operator methods from foundations to the cutting edge. The library is primarily intended for methods that directly relate to operator learning: new architectures, meta-algorithms, training methods and benchmark datasets. We are also interested in integrating interactive examples that showcase operator learning in action on small sample problems.

If your work provides one of the above, we would be thrilled to integrate it into the library. Otherwise, if your work simply relies on a version of the NeuralOperator codebase, we recommend publishing your code separately using a procedure outlined below.

Thank you for contributing to NeuralOperator! Your contributions help make this library better for the entire scientific machine learning community.


## Table of Contents

- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Development Guidelines](#development-guidelines)
- [Adding a New Model or Layer](#adding-a-new-model-or-layer)
- [Getting Help](#getting-help)
- [Publishing Code Built on the Library](#publishing-code-built-on-the-library)
- [License](#license)

&nbsp;

## Development Setup

### 1. Fork and Clone the Repository

1. **Fork the repository** on GitHub by clicking the "Fork" button

2. **Clone the library and connect your fork**:
   ```bash
   git clone https://github.com/neuraloperator/neuraloperator.git
   cd neuraloperator
   git remote rename origin upstream
   git remote add origin https://github.com/<YOUR_GIT_NAME>/neuraloperator.git
   git remote -v
   ```
   This should show:
   ```
   origin  https://github.com/<YOUR_GIT_NAME>/neuraloperator.git (fetch)
   origin  https://github.com/<YOUR_GIT_NAME>/neuraloperator.git (push)
   upstream        https://github.com/neuraloperator/neuraloperator.git (fetch)
   upstream        https://github.com/neuraloperator/neuraloperator.git (push)
   ```

### 2. Set Up Development Environment

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -e .[dev]
   # OR
   pip install neuraloperator[dev]
   ```


&nbsp;

## Development Workflow

### 1. Create a Branch and Make Changes

```bash
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
```

### 2. Write Tests

**Always add tests for new functionality!**

- **For models**: Add tests in `neuralop/models/tests/test_your_model.py`
- **For layers**: Add tests in `neuralop/layers/tests/test_your_layer.py`
- **For utilities**: Add tests in the appropriate test directory

**Example test structure:**
```python
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
```

### 3. Run Tests and Quality Checks

```bash
# Run all tests
pytest neuralop

# Run tests with verbose output
pytest neuralop -v

# Run specific test file
pytest neuralop/models/tests/test_your_model.py

# Format code
black .
```

### 4. Submit a Pull Request

Go to your fork on GitHub and click "Compare & pull request". 

Ensure your PR description clearly communicates what you have changed or added.

&nbsp;

## Development Guidelines

### Code Style
- **Follow PEP 8** style guidelines for Python code
- **Use meaningful names** for variables, functions, and classes
- **Write clear docstrings** using NumPy docstring format
- **Add type hints** where appropriate

### Code Formatting
Before submitting, ensure your code follows our style guide:

```bash
# Format with black
black .
```

Validate every update made by the ``black`` command

### Testing Requirements
- **All new code must have tests**
- **Run the full test suite** before submitting PRs
- **Use descriptive test names** that explain what is being tested
- **Test both normal operation and edge cases**
- **Aim for complete code coverage** for new functionality

### Documentation Standards
- **Update relevant documentation** when adding new features
- **Include clear examples** in docstrings
- **Add interactive examples** to the `examples/` directory when possible
- **Follow existing documentation style** and format
- **Include paper references** and mathematical formulations when relevant

### Building Documentation
The HTML documentation is built using Sphinx:

```bash
cd doc
make html
```

This builds the docs in `./doc/build/html`. Note that documentation requires additional dependencies from `./doc/requirements_doc.txt`.

To view documentation locally:
```bash
cd doc/build/html
python -m http.server 8000
# Visit http://localhost:8000
```

### Git Best Practices
- **Write clear, descriptive commit messages**
- **Keep commits focused and atomic** (one logical change per commit)
- **Rebase your branch** on main before submitting PRs
- **Use conventional commit format** when possible (feat:, fix:, docs:, etc.)

&nbsp;

## Adding a New Model or Layer

We welcome various types of contributions:

- **Bug fixes** - Report and fix bugs
- **New features** - Add new models, layers, or datasets
- **Examples** - Create new examples or improve existing ones
- **Performance** - Optimize existing code

### Adding a New Neural Operator Model

To add a new neural operator model:

1. **Create a new file** in `neuralop/models/your_model.py`
2. **Implement the model** as a subclass of `BaseModel`
3. **Add comprehensive tests** in `neuralop/models/tests/test_your_model.py`
4. **Update imports** in the appropriate `__init__.py` files
5. **Add documentation** with examples and mathematical formulations

**Example structure:**
```python
from neuralop.models import BaseModel

class YourModel(BaseModel):
    """Your model docstring here."""
    
    def __init__(self, ...):
        super().__init__()
        # Your implementation
    
    def forward(self, x):
        # Your forward pass
        return x
```

### Adding a New Layer

To add a new layer:

1. **Create a new file** in `neuralop/layers/your_layer.py`
2. **Ensure the layer** is a subclass of `torch.nn.Module`
3. **Add comprehensive tests** in `neuralop/layers/tests/test_your_layer.py`
4. **Update imports** in the appropriate `__init__.py` files

> **💡 Pro Tip**: For bonus points, add an interactive example featuring your new method to `./examples`. This helps both us and you: the simpler it is for new users to understand and adapt your method, the more visibility it will get!

&nbsp;

## Getting Help

### Before Asking for Help
1. **Check the documentation** and existing issues
2. **Search GitHub issues** for similar problems
3. **Provide a minimal reproducible example**
4. **Include error messages** and stack traces
5. **Specify your environment** (OS, Python version, PyTorch version)

### Contact Methods
- **GitHub Issues**: Create an issue for bugs or feature requests
- **GitHub Discussions**: Use for questions and ideas
- **Documentation**: Check the [official documentation](https://neuraloperator.github.io/)

&nbsp;

## Publishing Code Built on the Library

If you plan to use `neuralop` as the base of a project, we suggest the following workflow:

### 1. Set Up Clean Environment
```bash
python -m venv my_project_env
source my_project_env/bin/activate
```

### 2. Install NeuralOperator

**Option A: Latest PyPI release**
```bash
pip install neuralop
```

**Option B: From specific commit (for latest features)**
```bash
# Find commit hash from GitHub commit history
pip install git+https://github.com/neuraloperator/neuraloperator.git@<COMMIT_HASH>
```

### 3. Subclass Functionality

Create your own models by subclassing:

```python
from neuralop.models import FNO

class MyCustomFNO(FNO):
    def __init__(self, custom_param, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param
        # Add your custom layers here
    
    def forward(self, x):
        # Preprocessing
        x = self.custom_preprocessing(x)
        
        # Use the base FNO forward pass
        x = super().forward(x)
        
        # Postprocessing
        x = self.custom_postprocessing(x)
        
        return x
```

&nbsp;

---

## License

By contributing to NeuralOperator, you agree that your contributions will be licensed as described in the `LICENSE` file in the root directory of this source tree.
