# A simple function to dispatch nonlinearity modules. 
# This ensures that the nonlinearities used across modules in neuraloperator
# are parameterized while maintaining some standardization.
# This also allows us to store the nonlinearity kwarg as a string for serializing our models. 
from typing import Literal

from torch import nn
import torch.nn.functional as F

nonlinearity_modules = {'gelu': F.gelu,
                        'relu': F.relu,
                        'elu': F.elu,
                        'tanh': F.tanh,
                        'sigmoid': F.sigmoid,
                        'identity': nn.Identity()}

def get_nonlinearity(key: Literal["gelu", "relu", "elu", "sigmoid", "tanh", "identity"]):
    try:
        return nonlinearity_modules[key]
    except KeyError:
        raise NotImplementedError(f"tried to instantiate nonlinearity {key}, available nonlinearities are {list(nonlinearity_modules.keys())}")