from .spherical_swe import load_spherical_swe
from .pt_dataset import PTDataset
from .darcy import DarcyDataset, load_darcy_flow_small
from .burgers import load_burgers_1dtime
from .dict_dataset import DictDataset
from .data_transforms import DataProcessor
# only import MeshDataModule if open3d is built locally
from importlib.util import find_spec
if find_spec('open3d') is not None:
    from .mesh_datamodule import MeshDataModule
