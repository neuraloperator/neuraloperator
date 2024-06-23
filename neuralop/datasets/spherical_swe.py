from tensorly.utils import DefineDeprecated
from neuralop.data.datasets import spherical_swe

warning_msg = "Warning: neuralop.datasets.spherical_swe is deprecated and has been moved to neuralop.data.datasets.spherical_swe."
load_spherical_swe = DefineDeprecated(spherical_swe.load_spherical_swe, warning_msg)