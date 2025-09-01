import torch
"""
finite_diff.py implements utilities for computing derivatives via finite-difference method
"""

class FiniteDiff1D:
    """
    A comprehensive class for computing 1D finite differences with boundary handling.
    
    This class provides methods for computing derivatives of 1D fields using central finite differences.
    """
    
    def __init__(self, h=1.0, periodic_in_x=True):
        """
        Parameters
        ----------
        h : float, optional
            Grid spacing, by default 1.0
        periodic_in_x : bool, optional
            Whether to use periodic boundary conditions, by default True
        """
        self.h = h
        self.periodic_in_x = periodic_in_x
    
    def dx(self, u, order=1):
        """
        Compute derivative with respect to x.
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor
        order : int, optional
            Order of the derivative, by default 1
            
        Returns
        -------
        torch.Tensor
            Derivative with respect to x
        """
        if order == 1:
            return self._dx_1st(u)
        elif order == 2:
            return self._dx_2nd(u)
        else:
            raise ValueError("Only 1st and 2nd order derivatives currently supported")
    
    def _dx_1st(self, u):
        """First order derivative with respect to x."""
        
        if self.periodic_in_x:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i+1} - f_{i-1})/(2h)
            dx = (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / (2.0 * self.h)
            
        else:
            # Non-periodic case: handle boundaries separately
            dx = torch.zeros_like(u)
            
            # Interior points: Second-order central differences
            # (f_{i+1} - f_{i-1})/(2h)
            dx[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2.0 * self.h)
            
            # Left boundary: 3rd-order forward differences (-11f_{0} + 18f_{1} - 9f_{2} + 2f_{3})/(6h)
            dx[..., 0] = (-11*u[..., 0] + 18*u[..., 1] - 9*u[..., 2] + 2*u[..., 3]) / (6.0 * self.h)
            
            # Right boundary: 3rd-order backward differences (-2f_{n-4} + 9f_{n-3} - 18f_{n-2} + 11f_{n-1})/(6h)
            dx[..., -1] = (-2*u[..., -4] + 9*u[..., -3] - 18*u[..., -2] + 11*u[..., -1]) / (6.0 * self.h)
        
        return dx
    
    def _dx_2nd(self, u):
        """Second order derivative with respect to x."""
        
        if self.periodic_in_x:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i+1} - 2f_{i} + f_{i-1})/(h²)
            dxx = (torch.roll(u, -1, dims=-1) - 2*u + torch.roll(u, 1, dims=-1)) / (self.h**2)
        
        else:
            # Non-periodic case: handle boundaries separately
            dxx = torch.zeros_like(u)
            
            # Interior points: Second-order central differences
            # (f_{i+1} - 2f_{i} + f_{i-1})/(h²)
            dxx[..., 1:-1] = (u[..., 2:] - 2*u[..., 1:-1] + u[..., :-2]) / (self.h**2)
            
            # Boundary points: 3rd-order one-sided differences
            # Left boundary: 3rd-order forward differences (2f_{0} - 5f_{1} + 4f_{2} - f_{3})/h²
            dxx[..., 0] = (2*u[..., 0] - 5*u[..., 1] + 4*u[..., 2] - u[..., 3]) / (self.h**2)
            # Right boundary: 3rd-order backward differences (-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/h²
            dxx[..., -1] = (-u[..., -4] + 4*u[..., -3] - 5*u[..., -2] + 2*u[..., -1]) / (self.h**2)
        
        return dxx


class FiniteDiff2D:
    """
    A comprehensive class for computing 2D finite differences with boundary handling.
    
    This class provides methods for computing partial derivatives, gradients,
    divergence, curl, and Laplacian of 2D fields using central finite differences.
    """
    
    def __init__(self, h=(1.0, 1.0), periodic_in_x=True, periodic_in_y=True):
        """
        Parameters
        ----------
        h : tuple or float, optional
            Grid spacing for (y, x) directions, by default (1.0, 1.0)
        periodic_in_x : bool, optional
            Whether to use periodic boundary conditions in x-direction, by default True
        periodic_in_y : bool, optional
            Whether to use periodic boundary conditions in y-direction, by default True
        """
        if isinstance(h, float):
            self.h = (h, h)
        else:
            self.h = h
        self.periodic_in_x = periodic_in_x
        self.periodic_in_y = periodic_in_y
        
        # Validate parameters
        if len(self.h) != 2:
            raise ValueError("h must be a float or a tuple of length 2")


    def dx(self, u, order=1):
        """
        Compute partial derivative with respect to x.
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor
        order : int, optional
            Order of the derivative, by default 1
            
        Returns
        -------
        torch.Tensor
            Partial derivative with respect to x
        """
        if order == 1:
            return self._dx_1st(u)
        elif order == 2:
            return self._dx_2nd(u)
        else:
            raise ValueError("Only 1st and 2nd order derivatives currently supported")
    
    def dy(self, u, order=1):
        """
        Compute partial derivative with respect to y.
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor
        order : int, optional
            Order of the derivative, by default 1
            
        Returns
        -------
        torch.Tensor
            Partial derivative with respect to y
        """
        if order == 1:
            return self._dy_1st(u)
        elif order == 2:
            return self._dy_2nd(u)
        else:
            raise ValueError("Only 1st and 2nd order derivatives currently supported")
    
    
    def _dx_1st(self, u):
        """First order derivative with respect to x."""
        
        if self.periodic_in_x:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i+1,j} - f_{i-1,j})/(2h_{x})
            dx = (torch.roll(u, -1, dims=-2) - torch.roll(u, 1, dims=-2)) / (2.0 * self.h[0])
            
        else:
            # Non-periodic case: handle boundaries separately
            dx = torch.zeros_like(u)
            
            # Interior points: Second-order central differences
            # (f_{i+1,j} - f_{i-1,j})/(2h_{x})
            dx[..., 1:-1, :] = (u[..., 2:, :] - u[..., :-2, :]) / (2.0 * self.h[0])
            
            # Left boundary: 3rd-order forward differences (-11f_{0} + 18f_{1} - 9f_{2} + 2f_{3})/(6h_{x})
            dx[..., 0, :] = (-11*u[..., 0, :] + 18*u[..., 1, :] - 9*u[..., 2, :] + 2*u[..., 3, :]) / (6.0 * self.h[0])
            
            # Right boundary: 3rd-order backward differences (-2f_{n-4} + 9f_{n-3} - 18f_{n-2} + 11f_{n-1})/(6h_{x})
            dx[..., -1, :] = (-2*u[..., -4, :] + 9*u[..., -3, :] - 18*u[..., -2, :] + 11*u[..., -1, :]) / (6.0 * self.h[0])
        
        return dx
    
    
    def _dy_1st(self, u):
        """First order derivative with respect to y."""
        
        if self.periodic_in_y:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i,j+1} - f_{i,j-1})/(2h_{y})
            dy = (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / (2.0 * self.h[1])
            
        else:
            # Non-periodic case: handle boundaries separately
            dy = torch.zeros_like(u)
            
            # Interior points: Second-order central differences
            # (f_{i,j+1} - f_{i,j-1})/(2h_{y})
            dy[..., :, 1:-1] = (u[..., :, 2:] - u[..., :, :-2]) / (2.0 * self.h[1])
            
            # Bottom boundary: 3rd-order forward differences (-11f_{0} + 18f_{1} - 9f_{2} + 2f_{3})/(6h_{y})
            dy[..., :, 0] = (-11*u[..., :, 0] + 18*u[..., :, 1] - 9*u[..., :, 2] + 2*u[..., :, 3]) / (6.0 * self.h[1])
            
            # Top boundary: 3rd-order backward differences (-2f_{n-4} + 9f_{n-3} - 18f_{n-2} + 11f_{n-1})/(6h_{y})
            dy[..., :, -1] = (-2*u[..., :, -4] + 9*u[..., :, -3] - 18*u[..., :, -2] + 11*u[..., :, -1]) / (6.0 * self.h[1])
        
        return dy
    
    
    def _dx_2nd(self, u):
        """Second order derivative with respect to x."""
        
        if self.periodic_in_x:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i+1,j} - 2f_{i,j} + f_{i-1,j})/(h_{x}²)
            dxx = (torch.roll(u, -1, dims=-2) - 2*u + torch.roll(u, 1, dims=-2)) / (self.h[0]**2)
        
        else:
            # Non-periodic case: handle boundaries separately
            dxx = torch.zeros_like(u)
            
            # Interior points: Second-order central differences
            # (f_{i+1,j} - 2f_{i,j} + f_{i-1,j})/(h_{x}²)
            dxx[..., 1:-1, :] = (u[..., 2:, :] - 2*u[..., 1:-1, :] + u[..., :-2, :]) / (self.h[0]**2)
            
            # Boundary points: 3rd-order one-sided differences
            # Left boundary: 3rd-order forward differences (2f_{0} - 5f_{1} + 4f_{2} - f_{3})/h_{x}²
            dxx[..., 0, :] = (2*u[..., 0, :] - 5*u[..., 1, :] + 4*u[..., 2, :] - u[..., 3, :]) / (self.h[0]**2)
            # Right boundary: 3rd-order backward differences (-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/h_{x}²
            dxx[..., -1, :] = (-u[..., -4, :] + 4*u[..., -3, :] - 5*u[..., -2, :] + 2*u[..., -1, :]) / (self.h[0]**2)
        
        return dxx
    
    
    def _dy_2nd(self, u):
        """Second order derivative with respect to y."""
        
        if self.periodic_in_y:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i,j+1} - 2f_{i,j} + f_{i,j-1})/(h_{y}²)
            dyy = (torch.roll(u, -1, dims=-1) - 2*u + torch.roll(u, 1, dims=-1)) / (self.h[1]**2)
        
        else:
            # Non-periodic case: handle boundaries separately
            dyy = torch.zeros_like(u)
            
            # Interior points: Second-order central differences
            # (f_{i,j+1} - 2f_{i,j} + f_{i,j-1})/(h_{y}²)
            dyy[..., :, 1:-1] = (u[..., :, 2:] - 2*u[..., :, 1:-1] + u[..., :, :-2]) / (self.h[1]**2)
            
            # Boundary points: 3rd-order one-sided differences
            # Bottom boundary: 3rd-order forward differences (2f_{0} - 5f_{1} + 4f_{2} - f_{3})/h_{y}²
            dyy[..., :, 0] = (2*u[..., :, 0] - 5*u[..., :, 1] + 4*u[..., :, 2] - u[..., :, 3]) / (self.h[1]**2)
            # Top boundary: 3rd-order backward differences (-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/h_{y}²
            dyy[..., :, -1] = (-u[..., :, -4] + 4*u[..., :, -3] - 5*u[..., :, -2] + 2*u[..., :, -1]) / (self.h[1]**2)
        
        return dyy
    
    
    def laplacian(self, u):
        """
        Compute the Laplacian ∇²f.
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor
            
        Returns
        -------
        torch.Tensor
            The Laplacian of the input tensor
        """
        return self._dx_2nd(u) + self._dy_2nd(u)
    
    def divergence(self, u):
        """
        Compute the divergence ∇·u for 2D vector fields.
        
        Parameters
        ----------
        u : torch.Tensor
            Input vector field with shape (..., 2, height, width)
            
        Returns
        -------
        torch.Tensor
            The divergence of the vector field
        """
        if u.shape[-3] != 2:
            raise ValueError("Input must be a 2D vector field with 2 components")
        
        u1, u2 = u[..., 0, :, :], u[..., 1, :, :]
        return self.dx(u1) + self.dy(u2)
    
    
    def curl(self, u):
        """
        Compute the curl ∇×u for 2D vector fields.
        
        Parameters
        ----------
        u : torch.Tensor
            Input vector field with shape (..., 2, height, width)
            
        Returns
        -------
        torch.Tensor
            The curl of the vector field (scalar field in 2D)
        """
        if u.shape[-3] != 2:
            raise ValueError("Input must be a 2D vector field with 2 components")
        
        u1, u2 = u[..., 0, :, :], u[..., 1, :, :]
        return self.dx(u2) - self.dy(u1)
    
    
    def gradient(self, u):
        """
        Compute the gradient ∇f for scalar fields.
        
        Parameters
        ----------
        u : torch.Tensor
            Input scalar field
            
        Returns
        -------
        torch.Tensor
            The gradient of the scalar field with shape (..., 2, height, width)
        """
        grad_x = self.dx(u)
        grad_y = self.dy(u)
        
        return torch.stack([grad_x, grad_y], dim=-3)


class FiniteDiff3D:
    """
    A comprehensive class for computing 3D finite differences with boundary handling.
    
    This class provides methods for computing partial derivatives, gradients,
    divergence, curl, and Laplacian of 3D fields using central finite differences.
    """
    
    def __init__(self, h=(1.0, 1.0, 1.0), periodic_in_x=True, periodic_in_y=True, periodic_in_z=True):
        """
        Parameters
        ----------
        h : tuple or float, optional
            Grid spacing for (z, y, x) directions, by default (1.0, 1.0, 1.0)
        periodic_in_x : bool, optional
            Whether to use periodic boundary conditions in x-direction, by default True
        periodic_in_y : bool, optional
            Whether to use periodic boundary conditions in y-direction, by default True
        periodic_in_z : bool, optional
            Whether to use periodic boundary conditions in z-direction, by default True
        """
        if isinstance(h, float):
            self.h = (h, h, h)
        else:
            self.h = h
        self.periodic_in_x = periodic_in_x
        self.periodic_in_y = periodic_in_y
        self.periodic_in_z = periodic_in_z
        
        # Validate parameters
        if len(self.h) != 3:
            raise ValueError("h must be a float or a tuple of length 3")

    
    def dx(self, u, order=1):
        """
        Compute partial derivative with respect to x.
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor
        order : int, optional
            Order of the derivative, by default 1
            
        Returns
        -------
        torch.Tensor
            Partial derivative with respect to x
        """
        if order == 1:
            return self._dx_1st(u)
        elif order == 2:
            return self._dx_2nd(u)
        else:
            raise ValueError("Only 1st and 2nd order derivatives currently supported")
    
    def dy(self, u, order=1):
        """
        Compute partial derivative with respect to y.
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor
        order : int, optional
            Order of the derivative, by default 1
            
        Returns
        -------
        torch.Tensor
            Partial derivative with respect to y
        """
        if order == 1:
            return self._dy_1st(u)
        elif order == 2:
            return self._dy_2nd(u)
        else:
            raise ValueError("Only 1st and 2nd order derivatives currently supported")
    
    def dz(self, u, order=1):
        """
        Compute partial derivative with respect to z.
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor
        order : int, optional
            Order of the derivative, by default 1
            
        Returns
        -------
        torch.Tensor
            Partial derivative with respect to z
        """
        if order == 1:
            return self._dz_1st(u)
        elif order == 2:
            return self._dz_2nd(u)
        else:
            raise ValueError("Only 1st and 2nd order derivatives currently supported")
    
    
    def _dx_1st(self, u):
        """First order derivative with respect to x."""
        
        if self.periodic_in_x:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i+1,j,k} - f_{i-1,j,k})/(2h_{x})
            dx = (torch.roll(u, -1, dims=-3) - torch.roll(u, 1, dims=-3)) / (2.0 * self.h[0])
        
        else:
            # Non-periodic case: handle boundaries separately
            dx = torch.zeros_like(u)
            
            # Interior points: Second-order central differences
            # (f_{i+1,j,k} - f_{i-1,j,k})/(2h_{x})
            dx[..., 1:-1, :, :] = (u[..., 2:, :, :] - u[..., :-2, :, :]) / (2.0 * self.h[0])
            
            # Left boundary: 3rd-order forward differences (-11f_{0} + 18f_{1} - 9f_{2} + 2f_{3})/(6h_{x})
            dx[..., 0, :, :] = (-11*u[..., 0, :, :] + 18*u[..., 1, :, :] - 9*u[..., 2, :, :] + 2*u[..., 3, :, :]) / (6.0 * self.h[0])
            
            # Right boundary: 3rd-order backward differences (-2f_{n-4} + 9f_{n-3} - 18f_{n-2} + 11f_{n-1})/(6h_{x})
            dx[..., -1, :, :] = (-2*u[..., -4, :, :] + 9*u[..., -3, :, :] - 18*u[..., -2, :, :] + 11*u[..., -1, :, :]) / (6.0 * self.h[0])
        
        return dx
    
    
    def _dy_1st(self, u):
        """First order derivative with respect to y."""
        
        if self.periodic_in_y:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i,j+1,k} - f_{i,j-1,k})/(2h_{y})
            dy = (torch.roll(u, -1, dims=-2) - torch.roll(u, 1, dims=-2)) / (2.0 * self.h[1])
        
        else:
            # Non-periodic case: handle boundaries separately
            dy = torch.zeros_like(u)
            
            # Interior points: Second-order central differences
            # (f_{i,j+1,k} - f_{i,j-1,k})/(2h_{y})
            dy[..., :, 1:-1, :] = (u[..., :, 2:, :] - u[..., :, :-2, :]) / (2.0 * self.h[1])
            
            # Bottom boundary: 3rd-order forward differences (-11f_{0} + 18f_{1} - 9f_{2} + 2f_{3})/(6h_{y})
            dy[..., :, 0, :] = (-11*u[..., :, 0, :] + 18*u[..., :, 1, :] - 9*u[..., :, 2, :] + 2*u[..., :, 3, :]) / (6.0 * self.h[1])
            
            # Top boundary: 3rd-order backward differences (-2f_{n-4} + 9f_{n-3} - 18f_{n-2} + 11f_{n-1})/(6h_{y})
            dy[..., :, -1, :] = (-2*u[..., :, -4, :] + 9*u[..., :, -3, :] - 18*u[..., :, -2, :] + 11*u[..., :, -1, :]) / (6.0 * self.h[1])
        
        return dy
    
    def _dz_1st(self, u):
        """First order derivative with respect to z."""
        
        if self.periodic_in_z:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i,j,k+1} - f_{i,j,k-1})/(2h_{z})
            dz = (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / (2.0 * self.h[2])
        
        else:
            # Non-periodic case: handle boundaries separately
            dz = torch.zeros_like(u)
            
            # Interior points: Second-order central differences
            # (f_{i,j,k+1} - f_{i,j,k-1})/(2h_{z})
            dz[..., :, :, 1:-1] = (u[..., :, :, 2:] - u[..., :, :, :-2]) / (2.0 * self.h[2])
            
            # Front boundary: 3rd-order forward differences (-11f_{0} + 18f_{1} - 9f_{2} + 2f_{3})/(6h_{z})
            dz[..., :, :, 0] = (-11*u[..., :, :, 0] + 18*u[..., :, :, 1] - 9*u[..., :, :, 2] + 2*u[..., :, :, 3]) / (6.0 * self.h[2])
            
            # Back boundary: 3rd-order backward differences (-2f_{n-4} + 9f_{n-3} - 18f_{n-2} + 11f_{n-1})/(6h_{z})
            dz[..., :, :, -1] = (-2*u[..., :, :, -4] + 9*u[..., :, :, -3] - 18*u[..., :, :, -2] + 11*u[..., :, :, -1]) / (6.0 * self.h[2])
        
        return dz
    
    
    def _dx_2nd(self, u):
        """Second order derivative with respect to x."""
        
        if self.periodic_in_x:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i+1,j,k} - 2f_{i,j,k} + f_{i-1,j,k})/(h_{x}²)
            dxx = (torch.roll(u, -1, dims=-3) - 2*u + torch.roll(u, 1, dims=-3)) / (self.h[0]**2)
        
        else:
            # Non-periodic case: handle boundaries separately
            dxx = torch.zeros_like(u)
            
            # Interior points: Second-order central differences
            # (f_{i+1,j,k} - 2f_{i,j,k} + f_{i-1,j,k})/(h_{x}²)
            dxx[..., 1:-1, :, :] = (u[..., 2:, :, :] - 2*u[..., 1:-1, :, :] + u[..., :-2, :, :]) / (self.h[0]**2)
            
            # Boundary points: 3rd-order one-sided differences
            # Left boundary: 3rd-order forward differences (2f_{0} - 5f_{1} + 4f_{2} - f_{3})/h_{x}²
            dxx[..., 0, :, :] = (2*u[..., 0, :, :] - 5*u[..., 1, :, :] + 4*u[..., 2, :, :] - u[..., 3, :, :]) / (self.h[0]**2)
            # Right boundary: 3rd-order backward differences (-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/h_{x}²
            dxx[..., -1, :, :] = (-u[..., -4, :, :] + 4*u[..., -3, :, :] - 5*u[..., -2, :, :] + 2*u[..., -1, :, :]) / (self.h[0]**2)
        
        return dxx
    
    
    def _dy_2nd(self, u):
        """Second order derivative with respect to y."""
        
        if self.periodic_in_y:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i,j+1,k} - 2f_{i,j,k} + f_{i,j-1,k})/(h_{y}²)
            dyy = (torch.roll(u, -1, dims=-2) - 2*u + torch.roll(u, 1, dims=-2)) / (self.h[1]**2)
        
        else:
            # Non-periodic case: handle boundaries separately
            dyy = torch.zeros_like(u)
            
            # Interior points: Second-order central differences
            # (f_{i,j+1,k} - 2f_{i,j,k} + f_{i,j-1,k})/(h_{y}²)
            dyy[..., :, 1:-1, :] = (u[..., :, 2:, :] - 2*u[..., :, 1:-1, :] + u[..., :, :-2, :]) / (self.h[1]**2)
            
            # Boundary points: 3rd-order one-sided differences
            # Bottom boundary: 3rd-order forward differences (2f_{0} - 5f_{1} + 4f_{2} - f_{3})/h_{y}²
            dyy[..., :, 0, :] = (2*u[..., :, 0, :] - 5*u[..., :, 1, :] + 4*u[..., :, 2, :] - u[..., :, 3, :]) / (self.h[1]**2)
            # Top boundary: 3rd-order backward differences (-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/h_{y}²
            dyy[..., :, -1, :] = (-u[..., :, -4, :] + 4*u[..., :, -3, :] - 5*u[..., :, -2, :] + 2*u[..., :, -1, :]) / (self.h[1]**2)
        
        return dyy
    
    
    def _dz_2nd(self, u):
        """Second order derivative with respect to z."""
        
        if self.periodic_in_z:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i,j,k+1} - 2f_{i,j,k} + f_{i,j,k-1})/(h_{z}²)
            dzz = (torch.roll(u, -1, dims=-1) - 2*u + torch.roll(u, 1, dims=-1)) / (self.h[2]**2)
        
        else:
            # Non-periodic case: handle boundaries separately
            dzz = torch.zeros_like(u)
            
            # Interior points: Second-order central differences
            # (f_{i,j,k+1} - 2f_{i,j,k} + f_{i,j,k-1})/(h_{z}²)
            dzz[..., :, :, 1:-1] = (u[..., :, :, 2:] - 2*u[..., :, :, 1:-1] + u[..., :, :, :-2]) / (self.h[2]**2)
            
            # Boundary points: 3rd-order one-sided differences
            # Front boundary: 3rd-order forward differences (2f_{0} - 5f_{1} + 4f_{2} - f_{3})/h_{z}²
            dzz[..., :, :, 0] = (2*u[..., :, :, 0] - 5*u[..., :, :, 1] + 4*u[..., :, :, 2] - u[..., :, :, 3]) / (self.h[2]**2)
            # Back boundary: 3rd-order backward differences (-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/h_{z}²
            dzz[..., :, :, -1] = (-u[..., :, :, -4] + 4*u[..., :, :, -3] - 5*u[..., :, :, -2] + 2*u[..., :, :, -1]) / (self.h[2]**2)
        
        return dzz
    
    
    def laplacian(self, u):
        """
        Compute the Laplacian ∇²f.
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor
            
        Returns
        -------
        torch.Tensor
            The Laplacian of the input tensor
        """
        return self._dx_2nd(u) + self._dy_2nd(u) + self._dz_2nd(u)
    
    
    def divergence(self, u):
        """
        Compute the divergence ∇·u for 3D vector fields.
        
        Parameters
        ----------
        u : torch.Tensor
            Input vector field with shape (..., 3, depth, height, width)
            
        Returns
        -------
        torch.Tensor
            The divergence of the vector field
        """
        if u.shape[-4] != 3:
            raise ValueError("Input must be a 3D vector field with 3 components")
        
        u1, u2, u3 = u[..., 0, :, :, :], u[..., 1, :, :, :], u[..., 2, :, :, :]
        return self.dx(u1) + self.dy(u2) + self.dz(u3)
    
    
    def curl(self, u):
        """
        Compute the curl ∇×u for 3D vector fields.
        
        Parameters
        ----------
        u : torch.Tensor
            Input vector field with shape (..., 3, depth, height, width)
            
        Returns
        -------
        torch.Tensor
            The curl of the vector field with shape (..., 3, depth, height, width)
        """
        if u.shape[-4] != 3:
            raise ValueError("Input must be a 3D vector field with 3 components")
        
        u1, u2, u3 = u[..., 0, :, :, :], u[..., 1, :, :, :], u[..., 2, :, :, :]
        
        curl_x = self.dy(u3) - self.dz(u2)
        curl_y = self.dz(u1) - self.dx(u3)
        curl_z = self.dx(u2) - self.dy(u1)
        
        return torch.stack([curl_x, curl_y, curl_z], dim=-4)
    
    
    def gradient(self, u):
        """
        Compute the gradient ∇f for scalar fields.
        
        Parameters
        ----------
        u : torch.Tensor
            Input scalar field
            
        Returns
        -------
        torch.Tensor
            The gradient of the scalar field with shape (..., 3, depth, height, width)
        """
        grad_x = self.dx(u)
        grad_y = self.dy(u)
        grad_z = self.dz(u)
        
        return torch.stack([grad_x, grad_y, grad_z], dim=-4)




# Backward compatibility functions
def central_diff_1d(x, h, periodic_in_x=True):
    """
    Backward compatibility function for central_diff_1d.
    Creates a FiniteDiff1D instance and returns dx.
    """
    fd1d = FiniteDiff1D(h=h, periodic_in_x=periodic_in_x)
    return fd1d.dx(x)


def central_diff_2d(x, h, periodic_in_x=True, periodic_in_y=True):
    """
    Backward compatibility function for central_diff_2d.
    Creates a FiniteDiff2D instance and returns dx, dy.
    """
    fd2d = FiniteDiff2D(h=h, periodic_in_x=periodic_in_x, periodic_in_y=periodic_in_y)
    return fd2d.dx(x), fd2d.dy(x)


def central_diff_3d(x, h, periodic_in_x=True, periodic_in_y=True, periodic_in_z=True):
    """
    Backward compatibility function for central_diff_3d.
    Creates a FiniteDiff3D instance and returns dx, dy, dz.
    """
    fd3d = FiniteDiff3D(h=h, periodic_in_x=periodic_in_x, periodic_in_y=periodic_in_y, periodic_in_z=periodic_in_z)
    return fd3d.dx(x), fd3d.dy(x), fd3d.dz(x)



def get_non_uniform_fd_weights(points, num_neighbors=5, derivative_indices=[0], radius=None, regularize_lstsq=False):
    """
    Compute finite difference weights for approximating the first order derivative
    on an unstructured grid of points
    Parameters:
    -----------
    points : torch tensor of shape (N, d) containing the d coordinates of the N points
    num_neighbors: int for the number of nearest neighbors to include in the stencil (including the point itself)
                    At least 3 and at most N
    derivative_indices : indices of the derivatives to compute, e.g. in 2D, [0] for x, [1] for y, [0, 1] for x and y
    radius : float, the cutoff distance to use a neighbor as radius
             Neighbors with distances beyond this value are not used (their weights are set to zero)
             Always keep at least 3 neighbors (including the point itself)
    regularize_lstsq : bool, whether to regularize the least squares system
                        Sometimes torch.linalg.lstsq(A, b).solution creates artifacts so can add regularizer
                        But regularizer can deteriorate performance when system is well-conditioned
    
    Returns:
    --------
    indices : torch tensor of shape (N, k) for the indices of k nearest neighbors (including the point itself)
    fd_weights : torch tensor of weights of shape (N, len(derivative_indices), k)
                fd_weights[i,j,m] contains the weights for the m-th nearest neighbor 
                                        in the j-th 1st order derivative for the i-th point
    """

    N = points.shape[0]
    d = points.shape[1]
    k = min(max(num_neighbors, 3), N)  

    # Get the indices of the k nearest neighbors (including the point itself)
    pairwise_distances = torch.cdist(points, points, p=2)
    distances, indices = torch.topk(pairwise_distances, k=k, dim=1, largest=False)
    
    # Get mask for neighbors within cutoff radius (and always keep at least 3)
    if radius is None:
        radius_mask = torch.ones_like(distances, dtype=torch.bool)
    else:
        radius_mask = distances <= radius
        radius_mask[:, :3] = True

    # Initialize A to 1 since first row for each point and derivative is 1
    A = torch.ones((N, d+1, k), dtype=points.dtype, device=points.device)
    # Compute coordinate differences
    for i in range(d):
        A[:, i+1, :] = points[indices, i] - points[:, i].unsqueeze(1)
    # Repeat it for each derivative to be evaluated so it becomes of shape (N, len(derivative_indices), d+1, k)
    A = A.unsqueeze(1).expand(-1, len(derivative_indices), -1, -1)

    # Zero out columns for neighbors that are not within the radius
    A = A * radius_mask.unsqueeze(1).unsqueeze(2)
    
    # Compute right hand side
    b = torch.zeros((len(derivative_indices), d+1, 1), dtype=points.dtype, device=points.device)
    for i in range(len(derivative_indices)):
        b[i, derivative_indices[i]+1] = 1
    # Repeat so it becomes (N, len(derivative_indices), d+1, 1)
    b = b.unsqueeze(0).expand(N, -1, -1, -1)

    # Solve least squares system Aw = b  
    #    sometimes torch.linalg.lstsq(A, b).solution creates artifacts so can add regularizer
    #    but regularizer can deteriorate performance when system is well-conditioned

    if regularize_lstsq:

        lambda_reg = 1e-6
        I_k = torch.eye(k, dtype=A.dtype, device=A.device).unsqueeze(0).unsqueeze(0)

        AT = A.transpose(-2, -1)
        AT_b = torch.matmul(AT, b)
        AT_A = torch.matmul(AT, A) + lambda_reg * I_k

        # Use Cholesky decomposition to accelerate torch.linalg.solve(AT_A, AT_b).squeeze(-1) 
        fd_weights = torch.cholesky_solve(AT_b, torch.linalg.cholesky(AT_A)).squeeze(-1)  

    else:
        fd_weights = torch.linalg.lstsq(A, b).solution 

    return indices, fd_weights.squeeze(-1)


def non_uniform_fd(points, values, num_neighbors=5, derivative_indices=[0], radius=None, regularize_lstsq=False):
    """
    Compute finite difference approximation of the first order derivative on an unstructured grid of points
    Parameters:
    -----------
    points : torch tensor of shape (N, d) containing the d coordinates of the N points
    values : torch tensor of shape (N) containing the values of the function at the N points
    radius : float, the cutoff distance to use a neighbor as radius
             Neighbors with distances beyond this value are not used (their weights are set to zero)
             Always keep at least 3 neighbors (including the point itself)
    num_neighbors: int for the number of nearest neighbors to include in the stencil (including the point itself)
    derivative_indices : indices of the derivatives to compute, e.g. in 2D, [0] for x, [1] for y, [0, 1] for x and y
    regularize_lstsq : bool, whether to regularize the least squares system
                        Sometimes torch.linalg.lstsq(A, b).solution creates artifacts so can add regularizer
                        But regularizer can deteriorate performance when system is well-conditioned
    
    Returns:
    --------
    derivatives: tensor of shape (len(derivative_indices), N) of derivatives
            e.g. in 2D with derivative_indices=[0, 1], derivatives[0] is df(x,y)/dx and derivatives[1] is df(x,y)/dy
    
    """

    indices, fd_weights = get_non_uniform_fd_weights(points=points, 
                                                    num_neighbors=num_neighbors, 
                                                    derivative_indices=derivative_indices,
                                                    radius=radius,
                                                    regularize_lstsq=regularize_lstsq)

    derivatives = torch.einsum('nij,nj->in', fd_weights, values[indices])

    return derivatives
