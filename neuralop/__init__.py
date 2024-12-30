__version__ = '1.0.2'

from .models import TFNO3d, TFNO2d, TFNO1d, TFNO
from .models import get_model
from .data import datasets, transforms
from . import mpu
from .training import Trainer
from .losses import LpLoss, H1Loss, BurgersEqnLoss, ICLoss, WeightedSumLoss
