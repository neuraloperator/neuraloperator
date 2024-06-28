from tensorly.utils import DefineDeprecated

warning_msg = "Warning: neuralop.datasets.navier_stokes is deprecated and has been moved to neuralop.data.datasets.navier_stokes."
load_navier_stokes_pt = DefineDeprecated('neuralop.data.datasets.navier_stokes.load_navier_stokes_pt', warning_msg)