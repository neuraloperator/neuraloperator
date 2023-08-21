from ..resample import resample
import torch

def test_resample():
    a = torch.randn(10, 20, 40, 50)

    res_scale = [2, 3]
    axis = [-2, -1]

    b = resample(a, res_scale, axis)
    assert b.shape[-1] == 3*a.shape[-1] and b.shape[-2] == 2*a.shape[-2]

    a = torch.randn((10, 20, 40, 50, 60))

    res_scale = [0.5, 3,4]
    axis = [-3, -2, -1]
    b = resample(a, res_scale, axis)

    assert b.shape[-1] == 4*a.shape[-1] and b.shape[-2] == 3*a.shape[-2] and b.shape[-3] == int(0.5*a.shape[-3])