import numpy as np
import torch
import torch.nn.functional as F


def _resample_axis(x, new_res, axis):
    orig_dtype = x.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        x = x.to(torch.float32)

    axis = axis % x.ndim
    old_res = x.shape[axis]

    X = torch.fft.rfft(x, dim=axis, norm="forward")

    newshape = list(x.shape)
    newshape[axis] = new_res // 2 + 1
    Y = torch.zeros(newshape, dtype=X.dtype, device=x.device)

    modes = min(old_res, new_res)
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, modes // 2 + 1)
    modes_X = X[tuple(sl)]

    if modes % 2 == 0 and new_res != old_res:
        factor = 0.5 if new_res > old_res else 2.0
        sl_head = [slice(None)] * x.ndim
        sl_head[axis] = slice(0, modes // 2)
        sl_nyq = [slice(None)] * x.ndim
        sl_nyq[axis] = slice(modes // 2, modes // 2 + 1)
        modes_X = torch.cat(
            [modes_X[tuple(sl_head)], modes_X[tuple(sl_nyq)] * factor], dim=axis
        )

    Y[tuple(sl)] = modes_X

    y = torch.fft.irfft(Y, n=new_res, dim=axis, norm="forward")

    if orig_dtype in (torch.float16, torch.bfloat16):
        y = y.to(orig_dtype)

    return y


def resample(x, res_scale, axis, output_shape=None):
    """A module for generic n-dimentional interpolation.

    For 1D and 2D inputs, the ``resample`` function uses PyTorch’s built-in spatial interpolators
    for efficiency, applying linear interpolation for 1D data and bicubic interpolation for 2D data
    directly in the spatial domain.

    For 3D or higher-dimensional inputs, the ``resample`` function performs separable real Fourier
    interpolation sequentially along each specified axis, assuming periodic boundary conditions.
    When resizing along an axis where the minimum of the old and new resolution is even, the Nyquist
    frequency component is adjusted to preserve signal amplitude: for upsampling, the input Nyquist
    coefficient is scaled by 0.5 (as its energy splits across positive and negative frequencies in the
    larger spectrum); for downsampling, the output Nyquist coefficient is scaled by 2.0. Unchanged
    dimensions receive no adjustment.

    Parameters
    ----------
    x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
    res_scale: int or tuple
            Scaling factor along each of the dimensions in 'axis' parameter. If res_scale is scaler, then isotropic
            scaling is performed
    axis: axis or dimensions along which interpolation will be performed.
    output_shape : None or tuple[int]
    """

    if isinstance(res_scale, (float, int)):
        if axis is None:
            axis = list(range(2, x.ndim))
            res_scale = [res_scale] * len(axis)
        elif isinstance(axis, int):
            axis = [axis]
            res_scale = [res_scale]
        else:
            res_scale = [res_scale] * len(axis)
    else:
        assert len(res_scale) == len(axis), "leght of res_scale and axis are not same"

    if output_shape is None:
        new_size = tuple(
            [int(round(x.shape[a] * r)) for (a, r) in zip(axis, res_scale)]
        )
    else:
        new_size = tuple(output_shape)

    if len(axis) == 1:
        return F.interpolate(x, size=new_size[0], mode="linear", align_corners=True)
    if len(axis) == 2:
        return F.interpolate(x, size=new_size, mode="bicubic", align_corners=True)

    for a, s in zip(axis, new_size):
        x = _resample_axis(x, s, a)

    return x


def iterative_resample(x, res_scale, axis):
    if isinstance(axis, list) and isinstance(res_scale, (float, int)):
        res_scale = [res_scale] * len(axis)
    if not isinstance(axis, list) and isinstance(res_scale, list):
        raise Exception("Axis is not a list but Scale factors are")
    if (
        isinstance(axis, list)
        and isinstance(res_scale, list)
        and len(res_scale) != len(axis)
    ):
        raise Exception("Axis and Scal factor are in different sizes")

    if isinstance(axis, list):
        for i in range(len(res_scale)):
            rs = res_scale[i]
            a = axis[i]
            new_res = int(round(rs * x.shape[a]))
            x = _resample_axis(x, new_res, a)
        return x

    new_res = int(round(res_scale * x.shape[axis]))
    return _resample_axis(x, new_res, axis)
