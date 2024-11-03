from tensorly.utils import DefineDeprecated
from neuralop.data.datasets import tensor_dataset
from warnings import warn

warning_msg = "Warning: neuralop.datasets.tensor_dataset is deprecated and has been moved to neuralop.data.datasets.tensor_dataset."
warn(warning_msg)
TensorDataset = DefineDeprecated('neuralop.data.datasets.tensor_dataset.TensorDataset', tensor_dataset.TensorDataset)
