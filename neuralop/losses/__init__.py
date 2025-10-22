from .data_losses import LpLoss, H1Loss
from .equation_losses import BurgersEqnLoss, ICLoss
from .differentiation import FourierDiff, non_uniform_fd, FiniteDiff
from .meta_losses import (
    WeightedSumLoss,
    FieldwiseAggregatorLoss,
    Aggregator,
    Relobralo,
    SoftAdapt,
)
