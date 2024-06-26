from tensorly.utils import DefineDeprecated
from neuralop.data.datasets import zarr_dataset

warning_msg = "Warning: neuralop.datasets.zarr_dataset is deprecated and has been moved to neuralop.data.datasets.zarr_dataset."
ZarrDataset = DefineDeprecated(zarr_dataset.ZarrDataset, warning_msg)