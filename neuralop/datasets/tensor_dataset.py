from tensorly.utils import DefineDeprecated

warning_msg = "Warning: neuralop.datasets.tensor_dataset is deprecated and has been moved to neuralop.data.datasets.tensor_dataset."
TensorDataset = DefineDeprecated('neuralop.data.datasets.tensor_dataset.TensorDataset', warning_msg)