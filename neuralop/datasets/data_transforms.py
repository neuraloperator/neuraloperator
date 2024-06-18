from tensorly.utils import DefineDeprecated
from neuralop.data.transforms import data_processors

warning_msg = "Warning: neuralop.datasets.data_transforms is deprecated and has been moved to neuralop.data.datasets.data_processors."
DataProcessor = DefineDeprecated(data_processors.DataProcessor, warning_msg)
DefaultDataProcessor = DefineDeprecated(data_processors.DefaultDataProcessor, warning_msg)
IncrementalDataProcessor = DefineDeprecated(data_processors.IncrementalDataProcessor, warning_msg)
MGPatchingDataProcessor = DefineDeprecated(data_processors.MGPatchingDataProcessor, warning_msg)