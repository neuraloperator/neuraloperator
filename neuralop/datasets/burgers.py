from tensorly.utils import DefineDeprecated
from neuralop.data.datasets import burgers

warning_msg = "Warning: neuralop.datasets.burgers is deprecated and has been moved to neuralop.data.datasets.burgers."
load_burgers_1d = DefineDeprecated(burgers.load_burgers_1d, warning_msg)
load_burgers_1dtime = DefineDeprecated(burgers.load_burgers_1dtime, warning_msg)