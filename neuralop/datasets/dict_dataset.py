from tensorly.utils import DefineDeprecated
from neuralop.data.datasets import dict_dataset

warning_msg = "Warning: neuralop.datasets.dict_dataset is deprecated and has been moved to neuralop.data.datasets.dict_dataset."
DictDataset = DefineDeprecated(dict_dataset.DictDataset, warning_msg)