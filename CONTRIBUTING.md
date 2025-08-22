# Contributing to NeuralOperator

We welcome new contributions to the library! Our mission for NeuralOperator is to provide access to well-documented, robust implementations of neural operator methods from foundations to the cutting edge. The library is primarily intended for methods that directly relate to operator learning: new architectures, meta-algorithms, training methods and benchmark datasets. We are also interested in integrating interactive examples that showcase operator learning in action on small sample problems.

If your work provides one of the above, we would be thrilled to integrate it into the library. Otherwise, if your work simply relies on a version of the NeuralOperator codebase, we recommend publishing your code separately using a procedure outlined below.

## Table of Contents

- [Getting Started](#getting-started)
- [Extending NeuralOperator](#extending-neuraloperator)
- [Submitting Contributions](#submitting-contributions)
- [Publishing Code Built on the Library](#publishing-code-built-on-the-library)
- [Development Guidelines](#development-guidelines)
- [Debugging and Troubleshooting](#debugging-and-troubleshooting)
- [Getting Help](#getting-help)
- [License](#license)





## Getting Started



### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/neuraloperator.git
   cd neuraloperator
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/neuraloperator/neuraloperator.git
   ```
4. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
5. **Install development dependencies**:
   ```bash
   pip install -e .[dev]
   # OR
   pip install neuraloperator[dev]
   ```

First, ensure you have the latest version of the library installed for development, including requirements for the documentation. See the [installation guide](https://neuraloperator.github.io/dev/install.html) for more details.


## Extending NeuralOperator

We welcome various types of contributions:

- **Bug fixes** - Report and fix bugs
- **New features** - Add new models, layers, or datasets
- **Examples** - Create new examples or improve existing ones
- **Performance** - Optimize existing code

### Adding a New Neural Operator Model

To add a new neural operator model:

1. Create a new file in `neuralop/models/`.
2. Implement the model as a subclass of `BaseModel`.
3. Add a parametrized unit test file in `neuralop/models/tests`.
4. Update the `__init__.py` files accordingly to import your new model.

### Adding a New Layer

To add a new layer:

1. Create a new file in `neuralop/layers/`.
2. Ensure the layer is a subclass of `torch.nn.Module`.
3. Add a parametrized unit test file in `neuralop/layers/tests`.
4. Update the `__init__.py` files accordingly to import your new layer.

> **Note**: For optional bonus points, add an interactive example featuring your new method to `./examples`. This helps both us and you: the simpler it is for new users to understand and adapt your method, the more visibility it will get!

## Submitting Contributions

Follow these steps when making contributions:

1. **Create a new branch** for your feature or bug fix:
   ```bash
   git checkout -b feature-branch
   ```

2. **Write clean, well-documented code** following the library's coding standards.

3. **Add or update tests** in the appropriate directory. For instance, if your feature adds a model in `neuralop/models/mymodel.py`, you would add tests to `neuralop/models/tests/test_mymodel.py`.

4. **Run the test suite**:
   ```bash
   pytest neuralop
   ```

5. **Submit a pull request (PR)** on GitHub from your branch to the upstream `origin/main`. Ensure your PR description clearly communicates what you have changed or added.

## Publishing Code Built on the Library

If you plan to use `neuralop` as the base of a project, we suggest the following workflow:

1. **Set up a clean virtual environment**.

2. **Install `neuralop` via `pip`**. There are two ways to do this:

   - **Latest PyPI release**:
     ```bash
     pip install neuralop
     ```

   - **From a specific git commit hash** (if you need access to functionality added after the last PyPI release):
     
     Go to the repository's commit history page and locate the commit hash that corresponds to the repository state at which you want to install the repo. For most use cases, this will be the most recent commit.
     
     To find the commit hash, click the commit title, which will take you to the commit's URL. The hash will be the last component of the commit's URL, e.g., `https://github.com/neuraloperator/neuraloperator/commit/<COMMIT_HASH>`. Copy this hash to your clipboard.
     
     Then, use `pip` to install the library with the hash you just saved:
     ```bash
     pip install git+https://github.com/neuraloperator/neuraloperator.git@<COMMIT_HASH>
     ```

3. **Subclass functionality** when implementing new features. For instance, to create a modified `FNO` that performs extra steps during the forward pass:

   ```python
   from neuralop.models import FNO

   class MyFNO(FNO):
       def __init__(self, ...):
           super().__init__()

       def forward(self, x, ...):
           # do your special operations here
           x = my_operations(x, ...)
           
           # pass through the standard FNO.forward()
           x = super().forward(x, ...)
           
           # more operations could go here
           x = my_other_operations(x, ...)
           
           return x
   ```

## Development Guidelines

### Code Style
- Follow PEP 8 style guidelines for Python code
- Use meaningful and descriptive names for variables, functions, and classes
- Write clear comments and docstrings, using NumPy docstring format for functions and classes

### Code Formatting
Before you submit your changes, you should make sure your code adheres to our style-guide. The easiest way to do this is with ``black``:

```bash
black .
```

### Testing
- Ensure all new code has corresponding tests
- Run the full test suite before submitting PRs
- Use descriptive test names that explain what is being tested
- Test both normal operation and edge cases

To run the tests, simply run in the terminal:

```bash
pytest -v neuralop
```

### Documentation
- Update relevant documentation when adding new features
- Include clear examples in docstrings
- Consider adding interactive examples to the `examples/` directory
- Follow the existing documentation style and format
- Include paper references and mathematical formulations when relevant

### Building Documentation
The HTML for our documentation website is built using ``sphinx``. The documentation is built from inside the ``doc`` folder.

```bash
cd doc
make html
```

This will build the docs in ``./doc/build/html``.

Note that the documentation requires other dependencies installable from ``./doc/requirements_doc.txt``.

To view the documentation locally, run:

```bash
cd doc/build/html
python -m http.server [PORT_NUM]
```

The docs will then be viewable at ``localhost:PORT_NUM``.

### Git Best Practices
- Write clear, descriptive commit messages
- Keep commits focused and atomic
- Rebase your branch on main before submitting PRs


## Debugging and Troubleshooting

- Use `torch.set_detect_anomaly(True)` for debugging NaN gradients
- Check GPU memory usage with `nvidia-smi`
- Use `torch.autograd.profiler` for performance profiling
- Ensure dependencies are up to date with `pip list --outdated`

## Getting Help

For questions or issues:
- **GitHub Issues**: Create an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Documentation**: Check the [official documentation](https://neuraloperator.github.io/)

### Before Asking for Help
1. Check the documentation and existing issues
2. Provide a minimal reproducible example
3. Include error messages and stack traces
4. Specify your environment (OS, Python version, PyTorch version)

---

## License

By contributing to NeuralOperator, you agree that your contributions will be licensed as described in the `LICENSE` file in the root directory of this source tree.

---

## Acknowledgments

Thank you for contributing to NeuralOperator! Your contributions help make this library better for the entire scientific machine learning community.
