=============
API reference
=============

:mod:`neuralop`: Neural Operators in Python

.. automodule:: neuralop.models
    :no-members:
    :no-inherited-members:

.. _neuralop_models_ref:

Models
======

In `neuralop`, we provide a general Tensorized Fourier Neural Operator (TFNO) that supports most usecases.

We have a generic interface that works for any dimension, which is inferred based on `n_modes`
(a tuple with the number of modes to keep in the Fourier domain for each dimension.)

.. autosummary::
    :toctree: generated
    :template: class.rst

    TFNO

We also have dimension-specific classes:

.. autosummary::
    :toctree: generated
    :template: class.rst

    TFNO2d
    TFNO1d
    TFNO3d

Layers
======

In addition to the full architectures, we also provide building blocks:

.. automodule:: neuralop.models.fno_block
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    FactorizedSpectralConv

    FactorizedSpectralConv1d
    FactorizedSpectralConv2d
    FactorizedSpectralConv3d

Automatically apply resolution dependent domain padding: 

.. automodule:: neuralop.models.padding
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    DomainPadding

.. automodule:: neuralop.models.skip_connections
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    SoftGating

.. autosummary::
    :toctree: generated
    :template: function.rst

    skip_connection


Model Dispatching
=================
We provide a utility function to create model instances from a configuration.
It has the advantage of doing some checks on the parameters it receives.

.. automodule:: neuralop.models.model_dispatcher
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: function.rst

    get_model

Datasets
========

We ship a small dataset for testing:

.. automodule:: neuralop.datasets
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: function.rst

    load_darcy_flow_small
