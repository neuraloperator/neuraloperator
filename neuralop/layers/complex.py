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
        torch.cfloat
    )


def apply_complex(real_func, imag_func, x, dtype=torch.cfloat):
    """
    fr: a function (e.g., conv) to be applied on real part of x
    fi: a function (e.g., conv) to be applied on imag part of x
    x: complex input.
    """
    return (real_func(x.real) - imag_func(x.imag)).type(dtype) + 1j *\
          (real_func(x.imag) + imag_func(x.real)).type(
        dtype
    )

class ComplexValued(nn.Module):
    """
    Wrapper class that converts a standard nn.Module that operates on real data
    into a module that operates on complex-valued spatial data.
    """

    def __init__(self, module):
        super(ComplexValued, self).__init__()
        self.fr = deepcopy(module)
        self.fi = deepcopy(module)

    def forward(self, x):
        return apply_complex(self.fr, self.fi, x) 