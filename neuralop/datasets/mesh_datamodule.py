from tensorly.utils import DefineDeprecated
from neuralop.data.datasets import mesh_datamodule

warning_msg = "Warning: neuralop.datasets.mesh_datamodule is deprecated and has been moved to neuralop.data.datasets.mesh_datamodule."
MeshDataModule = DefineDeprecated(mesh_datamodule.MeshDataModule, warning_msg)