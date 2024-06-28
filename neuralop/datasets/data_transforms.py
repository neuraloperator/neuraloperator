from tensorly.utils import DefineDeprecated

warning_msg = "Warning: neuralop.datasets.data_transforms is deprecated and has been moved to neuralop.data.datasets.data_processors."
DataProcessor = DefineDeprecated('neuralop.data.transforms.data_processors.DataProcessor', warning_msg)
DefaultDataProcessor = DefineDeprecated('neuralop.data.transforms.data_processors.DefaultDataProcessor', warning_msg)
IncrementalDataProcessor = DefineDeprecated('neuralop.data.transforms.data_processors.IncrementalDataProcessor', warning_msg)
MGPatchingDataProcessor = DefineDeprecated('neuralop.data.transforms.data_processors.MGPatchingDataProcessor', warning_msg)