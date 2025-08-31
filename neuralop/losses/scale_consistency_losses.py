import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scale_no.data_augmentation import (
    RandomCropResize,
    RandomCropResizeTime,
    RandomCropResizeTimeAR,
)
from neuralop.data.transforms.rescale import (
    DarcyExtractBC,
    BurgersExtractBC,
    HelmholtzExtractBC,
)


import time


def LossSelfconsistency(
    model,
    x,
    loss_fn,
    y=None,
    re=None,
    rate=None,
    new_y=None,
    size_min=64,
    type="darcy",
    plot=False,
    group_action=None,
    align_corner=False,
):
    """
    Selfconsistency loss:

    Enforces that the model evaluated on the
    entire domain, and restricted to a subdomain
    be equal to the model directly evaluated on
    the subdomain.

    The subdomain is chosen randomly each time.
    """
    #

    batch_size = x.shape[0]

    if type == "darcy":
        ExtractBD = DarcyExtractBC
        transform_xy = RandomCropResize(p=1.0, size_min=size_min)
    elif type == "helmholtz":
        ExtractBD = HelmholtzExtractBC
        transform_xy = RandomCropResize(p=1.0, size_min=size_min)
    elif type == "NS":
        ExtractBD = lambda x, y: x  # No boundary
        transform_xy = RandomCropResizeTimeAR(p=1.0, size_min=size_min)
    elif type == "burgers":
        ExtractBD = lambda x, y: BurgersExtractBC(y)
        transform_xy = RandomCropResizeTime(p=1.0, size_min=size_min)
    else:
        print("boundary type not supported")

    if re == None:
        re = torch.ones(batch_size, 1, requires_grad=False).to(x.device)

    # If y is given, we use it as the ground truth. the gradient only flow to y_small_
    if y is not None:

        # resample on smaller domain
        i, j, h, w, re = transform_xy.get_params(x, y, re=re, rate=rate)

        x_small = transform_xy.crop(x, i, j, h, w)
        y_small = transform_xy.crop(y, i, j, h, w)
        x_small = ExtractBD(x_small, y_small)

        if group_action is not None:
            x_small, y_small = group_action(x_small, y_small)
        y_small_ = model(x_small, re)

        return loss_fn(y_small_, y_small)

    # If y is not given, we set y=model(x). We treat the subdomain as ground truth can detach y_small_
    else:
        mode = "sc"

        y = model(x, re)

        # resample on smaller domain
        i, j, h, w, re = transform_xy.get_params(x, y, re=re, rate=rate)
        if align_corner:
            i = 0
            j = 0

        x_small = transform_xy.crop(x, i, j, h, w)
        y_small = transform_xy.crop(y, i, j, h, w)
        x_small = ExtractBD(x_small, y_small)

        # if group_action is not None:
        #     x_small, y_small = group_action(x_small, y_small)
        y_small_ = model(x_small.detach(), re)

        if align_corner:
            H = int(y_small.shape[-2] // 2)
            W = int(y_small.shape[-1] // 2)
            y_small = y_small[..., :H, :W]
            y_small_ = y_small_[..., :H, :W]

        return loss_fn.truncated(y_small_, y_small)
