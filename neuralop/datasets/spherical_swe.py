from tensorly.utils import DefineDeprecated

warning_msg = "Warning: neuralop.datasets.spherical_swe is deprecated and has been moved to neuralop.data.datasets.spherical_swe."
load_spherical_swe = DefineDeprecated('neuralop.data.datasets.spherical_swe.load_spherical_swe', warning_msg)