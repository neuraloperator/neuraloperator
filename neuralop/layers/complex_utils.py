"""
Functionality for handling complex-valued spatial data
"""

from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F


def CGELU(x: torch.Tensor):
    """Complex GELU activation function
    Follows the formulation of CReLU from Deep Complex Networks (https://openreview.net/pdf?id=H1T2hmZAb)
    apply GELU is real and imag part of the input separately, then combine as complex number
    Args:
        x: complex tensor
    """

    return F.gelu(x.real).type(torch.cfloat) + 1j * F.gelu(x.imag).type(
        torch.cfloat
    )


def ctanh(x: torch.Tensor):
    """Complex-valued tanh stabilizer
    Apply ctanh is real and imag part of the input separately, then combine as complex number
    Args:
        x: complex tensor
    """
    return torch.tanh(x.real).type(torch.cfloat) + 1j * torch.tanh(x.imag).type(
        torch.cfloat()
    )


def apply_complex(fr, fi, x, dtype=torch.cfloat):
    return (fr(x.real) - fi(x.imag)).type(dtype) + 1j * (fr(x.imag) + fi(x.real)).type(
        dtype
    )


class ComplexValued(nn.Module):
    """
    Wrapper class that converts a standard nn.Module that operates on real data
    into a module that operates on complex-valued spatial data.
    """

    def __init__(self, mod):
        super(ComplexValued, self).__init__()
        self.fr = deepcopy(mod)
        self.fi = deepcopy(mod)

    def forward(self, x):
        return apply_complex(self.fr, self.fi, x) 