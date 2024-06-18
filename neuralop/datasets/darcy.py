from tensorly.utils import DefineDeprecated
from neuralop.data.datasets import darcy

warning_msg = "Warning: neuralop.datasets.darcy is deprecated and has been moved to neuralop.data.datasets.darcy."
load_darcy_flow_small = DefineDeprecated(darcy.load_darcy_flow_small, warning_msg)
load_darcy_pt = DefineDeprecated(darcy.load_darcy_pt, warning_msg)
