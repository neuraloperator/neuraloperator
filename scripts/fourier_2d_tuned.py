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


def compl_mul2d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixy,ioxy->boxy", a, b)

    # return torch.stack([
    #     op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
    #     op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    # ], dim=-1)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x, size=None):
        if size==None:
            size = x.size(-1)

        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2,3])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, size, size//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)


        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(size, size), dim=[2,3])
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, in_dim, out_dim, modes1, modes2, width):
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2

        self.width_list = [width*2//4, width*3//4, width*4//4, width*4//4, width*5//4]
        self.size_list = [64,] * 5
        self.grid_dim = 2

        self.fc0 = nn.Linear(in_dim+self.grid_dim, self.width_list[0])

        self.conv0 = SpectralConv2d(self.width_list[0]+self.grid_dim, self.width_list[1], self.modes1*4//4, self.modes2*4//4)
        self.conv1 = SpectralConv2d(self.width_list[1]+self.grid_dim, self.width_list[2], self.modes1*3//4, self.modes2*3//4)
        self.conv2 = SpectralConv2d(self.width_list[2]+self.grid_dim, self.width_list[3], self.modes1*2//4, self.modes2*2//4)
        self.conv3 = SpectralConv2d(self.width_list[3]+self.grid_dim, self.width_list[4], self.modes1*1//4, self.modes2*1//4)
        self.w0 = nn.Conv1d(self.width_list[0]+self.grid_dim, self.width_list[1], 1)
        self.w1 = nn.Conv1d(self.width_list[1]+self.grid_dim, self.width_list[2], 1)
        self.w2 = nn.Conv1d(self.width_list[2]+self.grid_dim, self.width_list[3], 1)
        self.w3 = nn.Conv1d(self.width_list[3]+self.grid_dim, self.width_list[4], 1)

        self.fc1 = nn.Linear(self.width_list[4], self.width_list[4]*2)
        self.fc2 = nn.Linear(self.width_list[4]*2, self.width_list[4]*2)
        self.fc3 = nn.Linear(self.width_list[4]*2, out_dim)

    def forward(self, x):

        batchsize = x.shape[0]
        size_x, size_y= x.shape[1], x.shape[2]
        grid = self.get_grid(size_x, batchsize, x.device)
        size_list = self.size_list

        x = torch.cat((x, grid.permute(0, 2, 3, 1).repeat([1,1,1,1])), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x = torch.cat((x, grid), dim=1)
        x1 = self.conv0(x, size_list[1])
        x2 = self.w0(x.view(batchsize, self.width_list[0]+self.grid_dim, size_list[0]**2)).view(batchsize, self.width_list[1], size_list[0], size_list[0])
        # x2 = F.interpolate(x2, size=size_list[1], mode='trilinear')
        x = x1 + x2
        x = F.selu(x)

        x = torch.cat((x, grid), dim=1)
        x1 = self.conv1(x, size_list[2])
        x2 = self.w1(x.view(batchsize, self.width_list[1]+self.grid_dim, size_list[1]**2)).view(batchsize, self.width_list[2], size_list[1], size_list[1])
        # x2 = F.interpolate(x2, size=size_list[2], mode='trilinear')
        x = x1 + x2
        x = F.selu(x)

        x = torch.cat((x, grid), dim=1)
        x1 = self.conv2(x, size_list[3])
        x2 = self.w2(x.view(batchsize, self.width_list[2]+self.grid_dim, size_list[2]**2)).view(batchsize, self.width_list[3], size_list[2], size_list[2])
        # x2 = F.interpolate(x2, size=size_list[3], mode='trilinear')
        x = x1 + x2
        x = F.selu(x)

        x = torch.cat((x, grid), dim=1)
        x1 = self.conv3(x, size_list[4])
        x2 = self.w3(x.view(batchsize, self.width_list[3]+self.grid_dim, size_list[3]**2)).view(batchsize, self.width_list[4], size_list[3], size_list[3])
        # x2 = F.interpolate(x2, size=size_list[4], mode='trilinear')
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.selu(x)
        x = self.fc2(x)
        x = F.selu(x)
        x = self.fc3(x)
        return x

    def get_grid(self, S, batchsize, device):
        gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
        gridx = gridx.reshape(1, 1, S, 1).repeat([batchsize, 1, 1, S])
        gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, S).repeat([batchsize, 1, S, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

class Net2d(nn.Module):
    def __init__(self, in_dim, out_dim, modes, width):
        super(Net2d, self).__init__()
        self.conv1 = SimpleBlock2d(in_dim, out_dim, modes, modes, width)

    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

# TRAIN_PATH = 'data/ns_data_V100_N400_T200_0.mat'
# TEST_PATH = 'data/ns_data_V100_N400_T200_0.mat'
# TRAIN_PATH = 'data/ns_data_V1000_N200_T400_0.mat'
# TEST_PATH = 'data/ns_data_V1000_N200_T400_0.mat'


ntrain = 20
ntest = 5

modes = 20
width = 32

in_dim = 4
out_dim = 2

batch_size = 5
batch_size2 = 5


epochs = 100
learning_rate = 0.0025
scheduler_step = 20
scheduler_gamma = 0.5

loss_k = 1
loss_group = False

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

path = 'KF_vel_N'+str(ntrain)+'_k' + str(loss_k)+'_g' + str(loss_group)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path


runtime = np.zeros(2, )
t1 = default_timer()


sub = 4
S = 64

T_in = 100
T = 400
T_out = T_in+T
step = 1



data = np.load('data/KFvelocity_Re40_N25_part1.npy')
data = torch.tensor(data, dtype=torch.float)
print(data.shape )

train_a = data[:ntrain,T_in-1:T_out-1,::sub,::sub,:].permute(0,2,3,1,4)
train_u = data[:ntrain,T_in:T_out,::sub,::sub,:].permute(0,2,3,1,4)

test_a = data[-ntest:,T_in-1:T_out-1,::sub,::sub,:].permute(0,2,3,1,4)
test_u = data[-ntest:,T_in:T_out,::sub,::sub,:].permute(0,2,3,1,4)

print(train_a.shape)
print(train_u.shape)
assert (S == train_u.shape[2])
assert (T == train_u.shape[3])



gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, 1, 1])
gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, 1, 1])

train_a = torch.cat((gridx.repeat([ntrain,1,1,T,1]), gridy.repeat([ntrain,1,1,T,1]), train_a), dim=-1)
test_a = torch.cat((gridx.repeat([ntest,1,1,T,1]), gridy.repeat([ntest,1,1,T,1]), test_a), dim=-1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

model = Net2d(in_dim, out_dim, modes, width).cuda()
# model = torch.load('model/KF_vel_N20_ep200_m12_w32')
# model = torch.load('model/KF_vol_500_N20_ep200_m12_w32')


print(model.count_params())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


lploss = LpLoss(size_average=False)
hsloss = HsLoss(k=loss_k, group=loss_group, size_average=False)

gridx = gridx.to(device)
gridy = gridy.to(device)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for xx, yy in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T):
            x = xx[:,:,:,t,:]
            y = yy[:,:,:,t]

            out = model(x)
            loss = hsloss(out.reshape(batch_size, S, S, out_dim), y.reshape(batch_size, S, S, out_dim))
            train_l2 += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_l2 = 0
    test_l2_hp = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T):
                x = xx[:, :, :, t, :]
                y = yy[:, :, :, t]

                out = model(x)
                test_l2 += lploss(out.reshape(batch_size, S, S, out_dim), y.reshape(batch_size, S, S, out_dim)).item()
                test_l2_hp += hsloss(out.reshape(batch_size, S, S, out_dim), y.reshape(batch_size, S, S, out_dim)).item()


    t2 = default_timer()
    scheduler.step()
    print(ep, t2 - t1, train_l2/ntrain/T, test_l2_hp/ntest/T, test_l2/ntest/T)
    # print(ep, t2 - t1, test_l2/ntest/T)


# torch.save(model, path_model)
#
#
# model.eval()
#
# test_a = test_a[0,:,:,0,-2:]
#
# T = 1000 - T_in
# pred = torch.zeros(S,S,T,2)
# gridx = gridx.reshape(1,S,S,1)
# gridy = gridy.reshape(1,S,S,1)
# x_out = test_a.reshape(1,S,S,2).cuda()
# with torch.no_grad():
#     for i in range(T):
#         print(i)
#         x_in = torch.cat([gridx, gridy, x_out], dim=-1)
#         x_out = model(x_in)
#         pred[:,:,i] = x_out.view(S,S,2)
#
#
# scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})




