__version__ = '0.1.0'

from .models import TFNO3d, TFNO2d, TFNO1d, TFNO
from .models import get_model
from . import data

from .training import Trainer
from .training import LpLoss, H1Loss

