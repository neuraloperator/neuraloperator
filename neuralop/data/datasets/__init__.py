from .darcy import DarcyDataset, load_darcy_flow_small
from .spherical_swe import load_spherical_swe
from .navier_stokes import NavierStokesDataset, load_navier_stokes_pt 
from .pt_dataset import PTDataset
from .burgers import BurgersDataset, load_burgers_1dtime
from .dict_dataset import DictDataset

# only import MeshDataModule if open3d is built locally
from importlib.util import find_spec
if find_spec('open3d') is not None:
    from .mesh_datamodule import MeshDataModule
