from tensorly.utils import DefineDeprecated

warning_msg = "Warning: neuralop.datasets.mesh_datamodule is deprecated and has been moved to neuralop.data.datasets.mesh_datamodule."
MeshDataModule = DefineDeprecated('neuralop.data.datasets.mesh_datamodule.MeshDataModule', warning_msg)