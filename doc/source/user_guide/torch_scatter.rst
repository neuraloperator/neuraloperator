.. _torch_scatter_guide:
Using torch-scatter
===================

We use the package ``torch_scatter`` (available on PyPI) to speed up neighborhood reductions during 
the final stage of the ``IntegralTransform`` layer, linked here: :ref:`gno_api`.

``torch_scatter`` uses compiled CUDA kernels 
to perform this reduction, providing both memory efficiency and speed advantages over a native PyTorch implementation.

However, since these kernels are precompiled, it is crucial to install the version of ``torch_scatter`` that is built for
your specific combination of OS, PyTorch and CUDA. All available wheels can be found `here <https://data.pyg.org/whl/>`_.  

.. note :: 
    The latest PyTorch version for which ``torch_scatter`` is built is 2.5.1, so you should downgrade if you are using 
    the latest (2.6.0) release. 