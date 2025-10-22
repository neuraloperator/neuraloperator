__version__ = "1.0.2"

from .models import TFNO, FNO, get_model
from .data import datasets, transforms
from . import mpu
from .training import Trainer
from .losses import (
    LpLoss,
    H1Loss,
    BurgersEqnLoss,
    ICLoss,
    WeightedSumLoss,
    Aggregator,
    Relobralo,
    SoftAdapt,
    FourierDiff,
    non_uniform_fd,
    FiniteDiff,
)
