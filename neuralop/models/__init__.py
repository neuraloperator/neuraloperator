from .fno import TFNO, TFNO1d, TFNO2d, TFNO3d
from .fno import FNO, FNO1d, FNO2d, FNO3d
# only import SFNO if torch_harmonics is built locally
try:
    from .sfno import SFNO
    from .local_no import LocalNO
except:
    pass
from .uno import UNO
from .uqno import UQNO
from .fnogno import FNOGNO
from .gino import GINO
from .base_model import get_model
