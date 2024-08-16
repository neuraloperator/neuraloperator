"""
SFNO - Spherical Fourier Neural Operator
Replaces the default SpectralConv (a convolution in the frequency domain 
over Fourier basis functions) with a SphericalConv (a convolution over the
spherical harmonic basis functions)
"""
from ..layers.spherical_convolution import SphericalConv
from .fno import FNO, partialclass

SFNO = partialclass("SFNO", FNO, factorization="dense", conv_module=SphericalConv)
SFNO.__doc__ = SFNO.__doc__.replace("Fourier", "Spherical Fourier", 1)
SFNO.__doc__ = SFNO.__doc__.replace("FNO", "SFNO")
SFNO.__doc__ = SFNO.__doc__.replace("fno", "sfno")