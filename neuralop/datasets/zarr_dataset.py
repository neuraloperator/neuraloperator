from tensorly.utils import DefineDeprecated

warning_msg = "Warning: neuralop.datasets.zarr_dataset is deprecated and has been moved to neuralop.data.datasets.zarr_dataset."
ZarrDataset = DefineDeprecated('neuralop.data.datasets.zarr_dataset.ZarrDataset', warning_msg)