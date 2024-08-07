import numpy as np
import torch
import scipy.io
from neuralop.losses import LpLoss
from pathlib import Path


class FC2D(object):
    # 2d Fourier Continuation helper

    def __init__(self, device, d=5, C=25):
    
        self.device = device
        self.d = d
        self.C = C
        self.A = torch.from_numpy(scipy.io.loadmat(Path(__file__).resolve().parent.joinpath( \
              "fc_data/A_d" + str(d) + "_C" + str(C) + ".mat"))['A']).double()
        self.Q = torch.from_numpy(scipy.io.loadmat(Path(__file__).resolve().parent.joinpath( \
              "fc_data/Q_d" + str(d) + "_C" + str(C) + ".mat"))['Q']).double()
        if device == 'cuda':
            self.A = self.A.cuda()
            self.Q = self.Q.cuda()

    def diff_x(self, u, domain_length_x = 1):

        nx = u.size(2)
        hx = domain_length_x / (nx - 1)
        fourPtsx = nx + self.C
        prdx = fourPtsx * hx
        u = u.double()

        if fourPtsx % 2 == 0:
                k_max = int(fourPtsx / 2)
                k_x = torch.cat((torch.arange(start = 0, end = k_max + 1, step = 1, device = self.device),
                    torch.arange(start = - k_max + 1, end = 0, step = 1, device = self.device)), 0)
        else:
                k_max = int((fourPtsx - 1) / 2)
                k_x = torch.cat((torch.arange(start = 0, end = k_max + 1, step = 1, device = self.device),
                    torch.arange(start = - k_max, end = 0, step = 1, device = self.device)), 0)
        der_coeffsx = 1j * 2.0 * np.pi / prdx * k_x

        # compute derivatives along the x-direction
        # First produce the periodic extension
        y1 = torch.einsum("hik,jk->hij", torch.einsum("hik,kj->hij", u[:, :, -self.d:], self.Q), self.A)
        y2 = torch.flip(torch.einsum("hik,jk->hij", torch.einsum("hik,kj->hij", torch.flip(u[:, :, :self.d], dims=(2,)), self.Q), self.A), dims=(2,))
        ucont = torch.cat([u,y1+y2], dim=2)
        uhat = torch.fft.fft(ucont, dim=2)
        uder = torch.fft.ifft(uhat * der_coeffsx).real	
        ux = uder[:, :, :nx].float()

        return ux

    def diff_xx(self, u, domain_length_x = 1):

        nx = u.size(2)
        hx = domain_length_x / (nx - 1)
        fourPtsx = nx + self.C
        prdx = fourPtsx * hx
        u = u.double()

        if fourPtsx % 2 == 0:
                k_max = int(fourPtsx / 2)
                k_x = torch.cat((torch.arange(start = 0, end = k_max + 1, step = 1, device = self.device),
                    torch.arange(start = - k_max + 1, end = 0, step = 1, device = self.device)), 0)
        else:
                k_max = int((fourPtsx - 1) / 2)
                k_x = torch.cat((torch.arange(start = 0, end = k_max + 1, step = 1, device = self.device),
                    torch.arange(start = - k_max, end = 0, step = 1, device = self.device)), 0)
        der_coeffsx = - 4.0 * np.pi * np.pi / prdx / prdx * k_x * k_x

        # compute derivatives along the x-direction
        # First produce the periodic extension
        y1 = torch.einsum("hik,jk->hij", torch.einsum("hik,kj->hij", u[:, :, -self.d:], self.Q), self.A)
        y2 = torch.flip(torch.einsum("hik,jk->hij", torch.einsum("hik,kj->hij", torch.flip(u[:, :, :self.d], dims=(2,)), self.Q), self.A), dims=(2,))
        # Compute the derivative of the extension
        ucont = torch.cat([u,y1+y2], dim=2)
        uhat = torch.fft.fft(ucont, dim=2)
        uder = torch.fft.ifft(uhat * der_coeffsx).real
        # Restrict to the original interval
        ux = uder[:, :, :nx].float()

        return ux        

    def diff_y(self, u, domain_length_y = 1):

        ny = u.size(1)
        nx = u.size(2)
        hy = domain_length_y / (ny - 1)
        fourPtsy = ny + self.C
        prdy = fourPtsy * hy
        u = u.double()

        if fourPtsy % 2 == 0:
                k_max = int(fourPtsy / 2)
                k_y = torch.cat((torch.arange(start = 0, end = k_max + 1, step = 1, device = self.device),
                    torch.arange(start = - k_max + 1, end = 0, step = 1, device = self.device)), 0).reshape(fourPtsy, 1).repeat(1, nx)
        else:
                k_max = int((fourPtsy - 1) / 2)
                k_y = torch.cat((torch.arange(start = 0, end = k_max + 1, step = 1, device = self.device),
                    torch.arange(start = - k_max, end = 0, step = 1, device = self.device)), 0).reshape(fourPtsy, 1).repeat(1, nx)	         

        der_coeffsy = 1j * 2.0 * np.pi / prdy * k_y

        # compute derivatives along the y-direction
        # First produce the periodic extension
        y1 = torch.einsum("ikl,jk->ijl", torch.einsum("ikl,kj->ijl", u[:, -self.d:, :], self.Q), self.A)
        y2 = torch.flip(torch.einsum("ikl,jk->ijl",torch.einsum("ikl,kj->ijl",torch.flip(u[:, :self.d, :], dims=(1,)), self.Q), self.A), dims=(1,))
        # Compute the derivative of the extension
        ucont = torch.cat([u,y1+y2], dim=1)
        uhat = torch.fft.fft(ucont, dim=1)
        uder = torch.fft.ifft(uhat * der_coeffsy, dim=1).real
        # Restrict to the original interval
        uy = uder[:, :ny, :].float()

        return uy