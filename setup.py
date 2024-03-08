try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

import re
from pathlib import Path

def version(root_path):
    """Returns the version taken from __init__.py

    Parameters
    ----------
    root_path : pathlib.Path
        path to the root of the package

    Reference
    ---------
    https://packaging.python.org/guides/single-sourcing-package-version/
    """
    version_path = root_path.joinpath('neuralop', '__init__.py')
    with version_path.open() as f:
        version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def readme(root_path):
    """Returns the text content of the README.rst of the package

    Parameters
    ----------
    root_path : pathlib.Path
        path to the root of the package
    """
    with root_path.joinpath('README.rst').open(encoding='UTF-8') as f:
        return f.read()


root_path = Path(__file__).parent
README = readme(root_path)
VERSION = version(root_path)


config = {
    'name': 'neuraloperator',
    'packages': find_packages(),
    'description': 'Learning (Tensorized) Neural Operators in PyTorch.',
    'long_description': README,
    'long_description_content_type' : 'text/x-rst',
    'authors': [
        {'name': "Jean Kossaifi", 'email': "jean.kossaifi@gmail.com"},
        {'name': "Nikola Kovachki", 'email': "nkovachki@caltech.edu"},
        {'name': "Zongyi Li", 'email': "zongyili@caltech.edu"}
        ],
    'version': VERSION,
    'install_requires': ['numpy', 'configmypy', 'pytest', 'black', 'tensorly', 'tensorly-torch', 'opt-einsum'],
    'license': 'Modified BSD',
    'scripts': [],
    'include_package_data': True,
    'package_data': {'': ['datasets/data/*.pt']},
    'classifiers': [
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3'
    ],
}

setup(**config)
