import torch
import torch.nn.functional as F

from torch.autograd import grad

from .differentiation import FiniteDiff


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

        # Get derivatives, du/dt and du/dx and d^2u/dxx
        fd2d = FiniteDiff(dim=2, h=(dt, dx), periodic_in_x=False, periodic_in_y=False)
        dudt, dudx = fd2d.dx(u), fd2d.dy(u)
        dudxx = fd2d.dy(u, order=2)

        # right hand side
        right_hand_side = -dudx * u + self.visc * dudxx

        # compute the loss of the left and right hand sides of Burgers' equation
        return self.loss(dudt, right_hand_side)

    def __call__(self, y_pred, **kwargs):
        if self.method == "fdm":
            return self.fdm(u=y_pred)
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


class PoissonInteriorLoss(object):
    """
    PoissonInteriorLoss computes the loss on the interior points of model outputs 
    according to Poisson's equation in 2d: ∇·((1 + 0.1u^2)∇u(x)) = f(x)

    Parameters
    ----------
    method : Literal['autograd'] only (for now)
        How to compute derivatives for equation loss. 
        
        * If 'autograd', differentiates using torch.autograd.grad. This can be used with outputs with any irregular
        point cloud structure.
    loss: Callable, optional
        Base loss class to compute distances between expected and true values, 
        by default torch.nn.functional.mse_loss
    """

    def __init__(self, method='autograd', loss=F.mse_loss):
        super().__init__()
        self.method = method
        self.loss = loss
    
    def autograd(self, u, output_queries, output_source_terms_domain, num_boundary, **kwargs):
        """
        Compute loss between the left-hand side and right-hand side of
        nonlinear Poisson's equation: ∇·((1 + 0.1u^2)∇u(x)) = f(x)

        u: torch.Tensor | dict
            output of the model. 

            * If output_queries is passed to the model as a dict, this will be a 
            dict of outputs provided over the points at each value in output_queries. 
            Each tensor will be shape (batch, n_points, 2).
            
            * If a tensor, u will be of shape (batch, num_boundary + num_interior, 2), where
            u[:, 0:num_boundary, :] are boundary points and u[:, num_boundary:, :] are interior points.
        output_queries: torch.Tensor | dict
            output queries provided to the model. If provided as a dict of tensors,
            u will also be returned as a dict keyed the same way. If provided as a tensor,
            u will be a tensor of the same shape except for number of channels. If a tensor,
            output_queries[:, 0:num_boundary, :] are boundary points and output_queries[:, num_boundary:, :] 
            are interior points.
        output_source_terms_domain: torch.Tensor
            source terms f(x) defined for this specific instance of Poisson's equation. 

        """  

        if isinstance(output_queries, dict):
            output_queries_domain = output_queries['domain']
            u_prime = grad(outputs=u.sum(), inputs=output_queries_domain,
                           create_graph=True, retain_graph=True)[0]
        else:
            #We only care about U defined over the interior. Grab it now if the entire U is passed.
            output_queries_domain = None
            u = u[:, num_boundary:, ...]
            u_prime = grad(outputs=u.sum(), inputs=output_queries,
                           create_graph=True, retain_graph=True)[0][:, num_boundary:, :]
        
        u_x = u_prime[:,:,0]
        u_y = u_prime[:,:,1]
        
        # compute second derivatives
        if output_queries_domain is not None:
            u_xx = grad(outputs=u_x.sum(), inputs=output_queries_domain, 
                        create_graph=True, retain_graph=True)[0][:, :, 0]
            u_yy = grad(outputs=u_y.sum(), inputs=output_queries_domain, 
                        create_graph=True, retain_graph=True)[0][:, :, 1]
        else:
            u_xx = grad(outputs=u_x.sum(), inputs=output_queries,
                        create_graph=True, retain_graph=True)[0][:, num_boundary:, 0]
            u_yy = grad(outputs=u_y.sum(), inputs=output_queries,
                        create_graph=True, retain_graph=True)[0][:, num_boundary:, 1]
        u_xx = u_xx.squeeze(0)
        u_yy = u_yy.squeeze(0)
        u_prime = u_prime.squeeze(0)
        u = u.squeeze([0, -1])

        # compute LHS of the Poisson equation
        u_sq = torch.pow(u, 2)
        laplacian = (u_xx + u_yy)
        norm_grad_u = torch.pow(u_prime, 2).sum(dim=-1)

        assert u_sq.shape == u_xx.shape == u_yy.shape == norm_grad_u.shape

        left_hand_side = laplacian + laplacian * 0.1 * u_sq + 0.2 * u * norm_grad_u
        output_source_terms_domain = output_source_terms_domain.squeeze(0)

        assert left_hand_side.shape == output_source_terms_domain.shape
        loss = self.loss(left_hand_side, output_source_terms_domain)

        assert not u_prime.isnan().any()
        assert not u_yy.isnan().any()
        assert not u_xx.isnan().any()
        del u_xx, u_yy, u_x, u_y, left_hand_side

        return loss
    
    def __call__(self, y_pred, **kwargs):
        if self.method == "autograd":
            return self.autograd(u=y_pred, **kwargs)
        elif self.method == "finite_difference":
            raise NotImplementedError()
        else:
            raise NotImplementedError()

class PoissonBoundaryLoss(object): 
    def __init__(self, loss=F.mse_loss):
        super().__init__()
        self.loss = loss
        self.counter = 0

    def __call__(self, y_pred, num_boundary, out_sub_level, y, output_queries, **kwargs):
        num_boundary = int(num_boundary.item() * out_sub_level)
        boundary_pred = y_pred.squeeze(0).squeeze(-1)[:num_boundary]
        y_bound = y.squeeze(0).squeeze(-1)[:num_boundary]
        
        assert boundary_pred.shape == y_bound.shape
        return self.loss(boundary_pred, y_bound)
    
class PoissonEqnLoss(object):
    """PoissonEqnLoss computes a weighted sum of equation loss computed on the interior points of a model's output
    and a boundary loss computed on the boundary points. 

    Parameters
    ----------
    boundary_weight : float
        weight by which to multiply boundary loss
    interior_weight : float
        weight by which to multiply interior loss
    diff_method : Literal['autograd', 'finite_difference'], optional
        method to use to compute derivatives, by default 'autograd'
    base_loss : Callable, optional
        base loss class to use inside equation and boundary loss, by default F.mse_loss
    """
    def __init__(self, boundary_weight, interior_weight, diff_method: str='autograd', base_loss=F.mse_loss): 
        super().__init__()
        self.boundary_weight = boundary_weight
        self.boundary_loss = PoissonBoundaryLoss(loss=base_loss)

        self.interior_weight = interior_weight
        self.interior_loss = PoissonInteriorLoss(method=diff_method, loss=base_loss)

    def __call__(self, out, y, **kwargs):
        if isinstance(out, dict):
            interior_loss = self.interior_weight * self.interior_loss(out['domain'], **kwargs)
            bc_loss = self.boundary_weight * self.boundary_loss(out['boundary'], y=y['boundary'],  **kwargs)
        else:
            interior_loss = self.interior_weight * self.interior_loss(out, **kwargs)
            bc_loss = self.boundary_weight * self.boundary_loss(out, y=y, **kwargs)
        return interior_loss + bc_loss
