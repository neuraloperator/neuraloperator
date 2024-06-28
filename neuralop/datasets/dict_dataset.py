from tensorly.utils import DefineDeprecated

warning_msg = "Warning: neuralop.datasets.dict_dataset is deprecated and has been moved to neuralop.data.datasets.dict_dataset."
DictDataset = DefineDeprecated('neuralop.data.datasets.dict_dataset.DictDataset', warning_msg)