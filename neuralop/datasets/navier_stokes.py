from tensorly.utils import DefineDeprecated
from neuralop.data.datasets import navier_stokes

warning_msg = "Warning: neuralop.datasets.navier_stokes is deprecated and has been moved to neuralop.data.datasets.navier_stokes."
load_navier_stokes_pt = DefineDeprecated(navier_stokes.load_navier_stokes_pt, warning_msg)