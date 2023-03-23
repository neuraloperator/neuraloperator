from models.fno_block import FNOBlocks
from models.spectral_convolution import FactorizedSpectralConv
import torch
from torchsummary import summary
from models.tfno import FNO
m = FNO([5,5],10)
#k = FNOBlocks(1,3, res_scaling=[0.5,0.5],n_modes = [5,5],n_layers=1)
#m = k(torch.randn(10,1,20,20))
#print(m.shape)
summary(m,(3,20,20))