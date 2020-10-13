import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

torch.manual_seed(0)
np.random.seed(0)


class Spectral_truncate(nn.Module):
    def __init__(self, modes1):
        super(Spectral_truncate, self).__init__()
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1

    def forward(self, x):
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 1, normalized=True, onesided=True)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(x_ft.shape, device=x.device)
        out_ft[:, :self.modes1, :] = x_ft[:, :self.modes1,  :]

        #Return to physical space
        x = torch.irfft(out_ft, 1, normalized=True, onesided=True, signal_sizes=(x.size(-1), ))
        return x





# TEST_PATH = 'data/burgers_v1000_t200_r1024_N2048.mat'
TEST_PATH = 'pred/burger_test.mat'

ntrain = 4800
ntest = 100


runtime = np.zeros(2, )
t1 = default_timer()


sub = 1



reader = MatReader(TEST_PATH)
test_u = reader.read_field('pred')[-ntest:,:]


# print(train_u.shape)
print(test_u.shape)



t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')



myloss = LpLoss(size_average=False)


test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_u, test_u), batch_size=1, shuffle=False)
with torch.no_grad():
    for i in range(0, 27):
        model = Spectral_truncate(i)
        test_l2 = 0
        test_l2_2 = 0
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(y)
            loss = myloss(out.view(1, -1), y.view(1, -1)).item()
            test_l2 = test_l2 + loss

            out2 = model(out)
            loss = myloss(out2.view(1, -1), out.view(1, -1)).item()
            test_l2_2 = test_l2_2 + loss

        print(i, test_l2 / ntest, test_l2_2 / ntest)





