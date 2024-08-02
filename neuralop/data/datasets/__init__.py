from .darcy import DarcyDataset, load_darcy_flow_small
from .navier_stokes import NavierStokesDataset, load_navier_stokes_pt 
from .pt_dataset import PTDataset
from .burgers import load_burgers_1dtime
from .dict_dataset import DictDataset

# only import MeshDataModule if open3d is built locally
try:
    from .mesh_datamodule import MeshDataModule
except ModuleNotFoundError:
    pass

# only import SphericalSWEDataset if torch_harmonics is built locally
try:
    from .spherical_swe import load_spherical_swe
except ModuleNotFoundError:
    pass
