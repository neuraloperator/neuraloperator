import torch
import torch.nn.functional as F

from .finite_diff import central_diff_2d
from .data_losses import LpLoss
from .utils import FC2D

class BurgersEqnLoss(object):
    """
    Computes loss for Burgers' equation.
    """

    def __init__(self, visc=0.01, method="fdm", loss=F.mse_loss, domain_length=1.0):
        super().__init__()
        self.visc = visc
        self.method = method
        self.loss = loss
        self.domain_length = domain_length
        if not isinstance(self.domain_length, (tuple, list)):
            self.domain_length = [self.domain_length] * 2

    def fdm(self, u):
        # remove extra channel dimensions
        u = u.squeeze(1)

        # shapes
        _, nt, nx = u.shape

        # we assume that the input is given on a regular grid
        dt = self.domain_length[0] / (nt - 1)
        dx = self.domain_length[1] / nx

        # du/dt and du/dx
        dudt, dudx = central_diff_2d(u, [dt, dx], fix_x_bnd=True, fix_y_bnd=True)

        # d^2u/dxx
        dudxx = (
            torch.roll(u, -1, dims=-1) - 2 * u + torch.roll(u, 1, dims=-1)
        ) / dx**2
        # fix boundary
        dudxx[..., 0] = (u[..., 2] - 2 * u[..., 1] + u[..., 0]) / dx**2
        dudxx[..., -1] = (u[..., -1] - 2 * u[..., -2] + u[..., -3]) / dx**2

        # right hand side
        right_hand_side = -dudx * u + self.visc * dudxx

        # compute the loss of the left and right hand sides of Burgers' equation
        return self.loss(dudt, right_hand_side)

    def __call__(self, y_pred, **kwargs):
        if self.method == "finite_difference":
            return self.fdm(u=y_pred)
        raise NotImplementedError()
    

class BCLoss(object):
    """
    Computes loss for boundary value problems.
    """

    def __init__(self, loss=F.mse_loss):
        super().__init__()
        self.loss = loss

    def boundary_condition_loss(self, u):
        u = u[:, 0, :, :] # squeeze 1-channel output
        size = u.shape[1]
        index_x = torch.cat([torch.tensor(range(0, size)), (size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)),
                         torch.zeros(size)], dim=0).long()
        index_y = torch.cat([(size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)), torch.zeros(size),
                            torch.tensor(range(0, size))], dim=0).long()

        boundary_u = u[:, index_x, index_y]
        truth_u = torch.zeros(boundary_u.shape, device=u.device)
        return self.loss(boundary_u, truth_u)

    def __call__(self, u, **kwargs):
        return self.boundary_condition_loss(u)

class DarcyEqnLoss(object):
    def __init__(self, method, device='cuda', d=5, C=25, sub=3):
        """
        # device indicates where the matrices A and Q necessary for FC should be stored
        # d and C are continuation parameters, and indicate which matrices A and Q to load
        """
        super().__init__()
        self.method = method
        self.sub = sub
        self.__name__ = 'eqn'
        if self.method == 'fourier_continuation':
            self.fc_helper = FC2D(device, d, C)

    def finite_difference(self, a, u, domain_length=1):
        # remove extra channel dimensions
        a = a[:, 0, :, :]
        
        u = u[:, 0, :, :]

        # compute the left hand side of the Darcy Flow equation
        # note: here we assume that the input is a regular grid
        n = u.size(1)
        dx = domain_length / (n - 1)
        dy = dx
        ux, uy = central_diff_2d(u, [dx,dy], fix_x_bnd=False, fix_y_bnd=False)
        a_ux = a * ux
        a_uy = a * uy

        a_uxx, _ = central_diff_2d(a_ux, [dx,dy], fix_x_bnd=True, fix_y_bnd=True)
        _, a_uyy = central_diff_2d(a_uy, [dx,dy], fix_x_bnd=True, fix_y_bnd=True)
       
        left_hand_side =  -(a_uxx + a_uyy)
        left_hand_side = left_hand_side[:, 2:-2, 2:-2]

        # compute the Lp loss of the left and right hand sides of the Darcy Flow equation
        forcing_fn = torch.ones(left_hand_side.shape, device=u.device)
        lploss = LpLoss(d=2, L=1., reductions='sum')
        loss = lploss(left_hand_side, forcing_fn)

        del ux, uy, a_ux, a_uy, a_uxx, a_uyy
        return loss
        
    def fourier_continuation(self, a, u, domain_length_x = 1, domain_length_y = 1):

        # remove extra channel dimensions
        u = u[:, 0, :, :]
        a = a[:, 0, :, :]

        # compute derivatives along the x-direction
        ux = self.fc.diff_x(u, domain_length_x)	

        # compute derivatives along the y-direction
        uy = self.fc.diff_y(u, domain_length_y)

        a_ux = a * ux
        a_uy = a * uy

        # compute derivatives along the x-direction
        a_uxx = self.fc_helper.diff_x(a_ux, domain_length_x)

        # compute derivatives along the y-direction
        a_uyy = self.fc_helper.diff_y(a_uy, domain_length_y)


        left_hand_side =  -(a_uxx + a_uyy)
        left_hand_side = left_hand_side[:, 2:-2, 2:-2] # Not necessary for FC, but can be done for purposes of comparison with FDM

        # compute the Lp loss of the left and right hand sides of the Darcy Flow equation
        forcing_fn = torch.ones(left_hand_side.shape, device=u.device)
        lploss = LpLoss(d=2, reductions='mean') # todo: size_average=True
        
        return lploss.rel(left_hand_side, forcing_fn)  
        # return lploss.rel(left_hand_side, forcing_fn), left_hand_side  

    def __call__(self, u, x, out_p=None, **kwargs):
        if self.method == 'finite_difference':
            return self.finite_difference(x, u)
        elif self.method == 'fourier_continuation':
            return self.fourier_continuation(x, u)
        else:
            raise NotImplementedError()

class ICLoss(object):
    """
    Computes loss for initial value problems.
    """

    def __init__(self, loss=F.mse_loss):
        super().__init__()
        self.loss = loss

    def initial_condition_loss(self, y_pred, x):
        boundary_true = x[:, 0, 0, :]
        boundary_pred = y_pred[:, 0, 0, :]
        return self.loss(boundary_pred, boundary_true)

    def __call__(self, y_pred, x, **kwargs):
        return self.initial_condition_loss(y_pred, x)
