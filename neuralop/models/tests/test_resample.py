from neuralop.models.fno_block import resample
import torch
def test_resample():
        a = torch.randn((10,20,40,50))

        res_scale = [2,3]
        axis = [-2,-1]
        b = resample(a,res_scale,axis)
        assert b.shape[-1] == 3*a.shape[-1] and b.shape[-2] == 2*a.shape[-2]