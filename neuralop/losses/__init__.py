from .data_losses import LpLoss, H1Loss
from .equation_losses import BurgersEqnLoss, ICLoss
from .meta_losses import WeightedSumLoss, FieldwiseAggregatorLoss, Aggregator, Relobralo, SoftAdapt
from .fourier_diff import fourier_derivative_1d, FourierDiff2D, FourierDiff3D
from .finite_diff import central_diff_1d, central_diff_2d, central_diff_3d, non_uniform_fd, FiniteDiff2D, FiniteDiff3D