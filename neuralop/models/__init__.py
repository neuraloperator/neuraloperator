from .fno import TFNO, TFNO1d, TFNO2d, TFNO3d
from .fno import FNO, FNO1d, FNO2d, FNO3d
# only import SFNO if torch_harmonics is built locally
from importlib.util import find_spec
if find_spec('torch_harmonics') is not None:
    from .sfno import SFNO
from .uno import UNO
from .fnogno import FNOGNO
from .gino import GINO
from .base_model import get_model
