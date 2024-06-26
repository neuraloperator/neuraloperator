from tensorly.utils import DefineDeprecated
from neuralop.data.datasets import pt_dataset

warning_msg = "Warning: neuralop.datasets.pt_dataset is deprecated and has been moved to neuralop.data.datasets.pt_dataset."
load_pt_traintestsplit = DefineDeprecated(pt_dataset.load_pt_traintestsplit, warning_msg)