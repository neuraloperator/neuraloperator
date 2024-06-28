from tensorly.utils import DefineDeprecated

warning_msg = "Warning: neuralop.datasets.burgers is deprecated and has been moved to neuralop.data.datasets.burgers."
load_burgers_1d = DefineDeprecated('neuralop.data.datasets.burgers.load_burgers_1d', warning_msg)
load_burgers_1dtime = DefineDeprecated('neuralop.data.datasets.burgers.load_burgers_1dtime', warning_msg)