from tensorly.utils import DefineDeprecated
from neuralop.data.datasets import tensor_dataset

warning_msg = "Warning: neuralop.datasets.tensor_dataset is deprecated and has been moved to neuralop.data.datasets.tensor_dataset."
TensorDataset = DefineDeprecated(tensor_dataset.TensorDataset, warning_msg)