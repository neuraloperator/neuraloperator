.. _api_ref:

#############
API reference
#############

:mod:`neuralop`: Neural Operators in Python

.. automodule:: neuralop.models
    :no-members:
    :no-inherited-members:

.. _neuralop_models_ref:

Models
=======

In :mod:`neuralop.models`, we provide neural operator models you can directly use on your applications.

FNO
----

We provide a general Fourier Neural Operator (TFNO) that supports most usecases.

We have a generic interface that works for any dimension, which is inferred based on `n_modes`
(a tuple with the number of modes to keep in the Fourier domain for each dimension.)

.. _fno_api:
.. autosummary::
    :toctree: generated
    :template: class.rst

    FNO

We also have dimension-specific classes:

.. autosummary::
    :toctree: generated
    :template: class.rst

    FNO1d
    FNO2d
    FNO3d

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensorized FNO (TFNO)
----------------------

N-D version: 
.. _tfno_api:

.. autosummary::
    :toctree: generated
    :template: class.rst

    TFNO

Dimension-specific classes:

.. autosummary::
    :toctree: generated
    :template: class.rst

    TFNO1d
    TFNO2d
    TFNO3d

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Spherical Fourier Neural Operators (SFNO)
--------------------------------------------
.. _sfno_api:

.. autosummary::
    :toctree: generated
    :template: class.rst

    SFNO

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Geometry-Informed Neural Operators (GINO)
------------------------------------------
.. _gino_api:

.. autosummary::
    :toctree: generated
    :template: class.rst

    GINO

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Local Neural Operators (LocalNO)
--------------------------------------------
.. _local_no_api:

.. autosummary::
    :toctree: generated
    :template: class.rst

    LocalNO


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

U-shaped Neural Operators (U-NO)
---------------------------------
.. _uno_api:

.. autosummary::
    :toctree: generated
    :template: class.rst
    
    UNO

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Layers
=======

.. automodule:: neuralop.layers
    :no-members:
    :no-inherited-members:

.. _neuralop_layers_ref:


In addition to the full architectures, we also provide 
in :mod:`neuralop.layers` building blocks,
in the form of PyTorch layers, that you can use to build your own models:

.. _spectral_conv_api:
Fourier Convolutions
---------------------
.. automodule:: neuralop.layers.spectral_convolution
    :no-members:
    :no-inherited-members:

General SpectralConv layer:

.. autosummary::
    :toctree: generated
    :template: class.rst

    SpectralConv

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _sfno_api:
Spherical Convolutions
-----------------------

.. automodule:: neuralop.layers.spherical_convolution
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    SphericalConv

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To support geometry-informed (GINO) models, we also offer the ability to integrate kernels in the spatial domain,
 which we formulate as mappings between arbitrary coordinate meshes.

.. _gno_api:
Graph convolutions and kernel integration
-----------------------------------------

.. automodule:: neuralop.layers.gno_block
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    GNOBlock

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: neuralop.layers.integral_transform
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    IntegralTransform

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also provide additional layers that implement standard deep learning architectures as neural operators.

**Local Integral/Differential Convolutions**

.. _diff_conv_api:

.. automodule:: neuralop.layers.differential_conv
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    FiniteDifferenceConvolution

**Discrete-Continuous (DISCO) Convolutions**

.. _disco_conv_api:

.. automodule:: neuralop.layers.discrete_continuous_convolution
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    DiscreteContinuousConv2d
    DiscreteContinuousConvTranspose2d
    EquidistantDiscreteContinuousConv2d
    EquidistantDiscreteContinuousConvTranspose2d

**Local NO Blocks**

.. automodule:: neuralop.layers.local_no_block
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    LocalNOBlocks

**Codomain Attention (Transformer) Blocks**

.. automodule:: neuralop.layers.coda_layer
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    CODALayer

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Embeddings**

Apply positional embeddings as additional channels on a function:

.. automodule:: neuralop.layers.embeddings
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    GridEmbeddingND
    GridEmbedding2D
    SinusoidalEmbedding

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Neighbor search**

Find neighborhoods on arbitrary coordinate meshes:

.. automodule:: neuralop.layers.neighbor_search
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    NeighborSearch

.. autosummary::
    :toctree: generated
    :template: function.rst

    native_neighbor_search

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Other resolution-invariant operations
-------------------------------------

Positional embedding layers:

.. automodule:: neuralop.layers.embeddings
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    GridEmbeddingND
    SinusoidalEmbedding

Automatically apply resolution dependent domain padding: 

.. automodule:: neuralop.layers.padding
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    DomainPadding

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: neuralop.layers.skip_connections
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

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Model Dispatching
===================
We provide a utility function to create model instances from a configuration.
It has the advantage of doing some checks on the parameters it receives.

.. automodule:: neuralop.models.base_model
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: function.rst

    get_model
    available_models

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training
=========
We provide functionality that automates the boilerplate code associated with 
training a machine learning model to minimize a loss function on a dataset:

.. automodule:: neuralop.training
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    Trainer
    IncrementalFNOTrainer

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: neuralop.losses
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    LpLoss
    H1Loss

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data
========
In `neuralop.data`, we provide APIs for standardizing PDE datasets (`.datasets`) and transforming raw data into model inputs (`.transforms`).

We also ship a small dataset for testing:

.. automodule:: neuralop.data.datasets
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: function.rst

    load_darcy_flow_small

We provide downloadable datasets for Darcy-Flow, Navier-Stokes, and Car-CFD, as well as a general-purpose tensor dataset. 

.. automodule:: neuralop.data.datasets.darcy
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    DarcyDataset

.. automodule:: neuralop.data.datasets.navier_stokes
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    NavierStokesDataset

.. automodule:: neuralop.data.datasets.car_cfd_dataset
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    CarCFDDataset

.. automodule:: neuralop.data.datasets.tensor_dataset
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    TensorDataset

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DataProcessors
--------------

Much like PyTorch's `Torchvision.Datasets` module, our `data` module also includes
utilities to transform data from its raw form into the form expected by models and 
loss functions:

.. automodule:: neuralop.data.transforms.data_processors
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: class.rst

    DefaultDataProcessor
    MGPatchingDataProcessor

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
