import torch
from ..layers.fourier_continuation import FCLegendre, FCGram

"""
differentiation.py implements utilities for computing derivatives via finite-differences
and Fourier/spectral differentiation methods
"""


class FiniteDiff:
    """
    A unified class for computing finite differences in 1D, 2D, or 3D.

    This class provides comprehensive methods for computing derivatives using finite differences
    with support for both periodic and non-periodic boundary conditions.

    Finite Difference Methods
    -------------------------
    The class implements high-order finite difference schemes:

    - Interior points: Second-order central differences for optimal accuracy
    - Periodic boundaries: Uses torch.roll for seamless periodic wrapping.
    - Non-periodic boundaries: Uses third-order one-sided differences at boundary points.

    Mathematical Formulas
    ---------------------
    For first-order derivatives:
    - Interior: (f_{i+1} - f_{i-1})/(2h)  [2nd order central]
    - Left boundary: (-11f_0 + 18f_1 - 9f_2 + 2f_3)/(6h)  [3rd order forward]
    - Right boundary: (-2f_{n-4} + 9f_{n-3} - 18f_{n-2} + 11f_{n-1})/(6h)  [3rd order backward]

    For second-order derivatives:
    - Interior: (f_{i+1} - 2f_i + f_{i-1})/(h²)  [2nd order central]
    - Left boundary: (2f_0 - 5f_1 + 4f_2 - f_3)/(h²)  [3rd order forward]
    - Right boundary: (-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/(h²)  [3rd order backward]

    Available Methods
    ----------------
    Derivative Methods:
    - dx(u, order=1): Compute derivative with respect to x
    - dy(u, order=1): Compute derivative with respect to y (2D/3D only)
    - dz(u, order=1): Compute derivative with respect to z (3D only)

    Vector Calculus Operators:
    - laplacian(u): Compute the Laplacian ∇²f
    - gradient(u): Compute the gradient ∇f (returns vector field)
    - divergence(u): Compute the divergence ∇·u (for vector fields)
    - curl(u): Compute the curl ∇×u (for vector fields, 2D/3D only)


    Examples
    --------
    >>> # 1D finite differences
    >>> x = torch.linspace(0, 2*torch.pi, 100)
    >>> u = torch.sin(x)
    >>> fd1d = FiniteDiff(dim=1, h=0.1, periodic_in_x=True)
    >>> du_dx = fd1d.dx(u)  # First derivative
    >>> d2u_dx2 = fd1d.dx(u, order=2)  # Second derivative
    >>>
    >>> # 2D finite differences
    >>> fd2d = FiniteDiff(dim=2, h=(0.1, 0.1), periodic_in_x=True, periodic_in_y=False)
    >>> x = torch.linspace(0, 2*torch.pi, 50)
    >>> y = torch.linspace(0, 2*torch.pi, 50)
    >>> X, Y = torch.meshgrid(x, y, indexing='ij')
    >>> u = torch.sin(X) * torch.cos(Y)
    >>> du_dx = fd2d.dx(u)
    >>> du_dy = fd2d.dy(u)
    >>> grad = fd2d.gradient(u)  # Returns [du_dx, du_dy]
    >>>
    >>> # 3D finite differences
    >>> fd3d = FiniteDiff(dim=3, h=(0.1, 0.1, 0.1), periodic_in_x=True, periodic_in_y=True, periodic_in_z=False)
    >>> x = torch.linspace(0, 2*torch.pi, 20)
    >>> y = torch.linspace(0, 2*torch.pi, 20)
    >>> z = torch.linspace(0, 2*torch.pi, 20)
    >>> X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    >>> u = torch.sin(X) * torch.cos(Y) * torch.sin(Z)  # 3D scalar field
    >>> du_dx = fd3d.dx(u)
    >>> du_dy = fd3d.dy(u)
    >>> du_dz = fd3d.dz(u)
    >>> laplacian = fd3d.laplacian(u)  # Sum of all second derivatives
    >>>
    >>> # Vector field operations
    >>> vx = torch.sin(X) * torch.cos(Y) * torch.sin(Z)
    >>> vy = torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    >>> vz = torch.sin(X) * torch.sin(Y) * torch.cos(Z)
    >>> v = torch.stack([vx, vy, vz], dim=-4)  # 3D vector field
    >>> div_v = fd3d.divergence(v)  # Scalar field
    >>> curl_v = fd3d.curl(v)  # Vector field

    """

    def __init__(
            self,
            dim,
            h=1.0,
            periodic_in_x=True,
            periodic_in_y=True,
            periodic_in_z=True):
        """
        Initialize the FiniteDiff class for computing finite differences.

        Parameters
        ----------
        dim : int
            Dimension of the input field. Must be 1, 2, or 3.
        h : float or tuple, optional
            Grid spacing(s) for finite difference calculations, by default 1.0.
            - For 1D: single float or tuple with one element
            - For 2D: tuple (h_x, h_y) or single float for uniform spacing
            - For 3D: tuple (h_x, h_y, h_z) or single float for uniform spacing
        periodic_in_x : bool, optional
            Whether to use periodic boundary conditions in x-direction, by default True.
            When True, uses torch.roll for efficient periodic wrapping.
            When False, uses high-order one-sided differences at boundaries.
        periodic_in_y : bool, optional
            Whether to use periodic boundary conditions in y-direction, by default True.
            When True, uses torch.roll for efficient periodic wrapping.
            When False, uses high-order one-sided differences at boundaries.
            Only used for 2D and 3D fields.
        periodic_in_z : bool, optional
            Whether to use periodic boundary conditions in z-direction, by default True.
            When True, uses torch.roll for efficient periodic wrapping.
            When False, uses high-order one-sided differences at boundaries.
            Only used for 3D fields.

        """

        # Check if dim is valid
        if dim not in [1, 2, 3]:
            raise ValueError("dim must be 1, 2, or 3")

        self.dim = dim

        # Set up grid spacing
        if isinstance(h, (int, float)):
            # Create tuple of length dim with repeated h value
            self.h = tuple(h for _ in range(dim))
        else:
            # h is already a tuple/list
            if len(h) != dim:
                raise ValueError(f"For {dim}D, h must be a float or a tuple of length {dim}")
            self.h = tuple(h)  # Convert to tuple

        # Set up periodic conditions
        self.periodic_in_x = periodic_in_x
        if dim >= 2:
            self.periodic_in_y = periodic_in_y
        if dim >= 3:
            self.periodic_in_z = periodic_in_z
    
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
        if self.dim == 1:
            return self._dx_1d(u, order)
        elif self.dim == 2:
            return self._dx_2d(u, order)
        else:  # dim == 3
            return self._dx_3d(u, order)

    def dy(self, u, order=1):
        """
        Compute derivative with respect to y.

        Parameters
        ----------
        u : torch.Tensor
            Input tensor
        order : int, optional
            Order of the derivative, by default 1

        Returns
        -------
        torch.Tensor
            Derivative with respect to y
        """
        if self.dim < 2:
            raise ValueError("dy is only available for 2D and 3D")
        elif self.dim == 2:
            return self._dy_2d(u, order)
        else:  # dim == 3
            return self._dy_3d(u, order)

    def dz(self, u, order=1):
        """
        Compute derivative with respect to z.

        Parameters
        ----------
        u : torch.Tensor
            Input tensor
        order : int, optional
            Order of the derivative, by default 1

        Returns
        -------
        torch.Tensor
            Derivative with respect to z
        """
        if self.dim < 3:
            raise ValueError("dz is only available for 3D")
        return self._dz_3d(u, order)

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
        if self.dim == 1:
            return self._dx_1d(u, 2)
        elif self.dim == 2:
            return self._dx_2d(u, 2) + self._dy_2d(u, 2)
        else:  # dim == 3
            return self._dx_3d(u, 2) + self._dy_3d(u, 2) + self._dz_3d(u, 2)

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
            The gradient of the scalar field
        """
        if self.dim == 1:
            return self._dx_1d(u, 1)
        elif self.dim == 2:
            grad_x = self._dx_2d(u, 1)
            grad_y = self._dy_2d(u, 1)
            return torch.stack([grad_x, grad_y], dim=-3)
        else:  # dim == 3
            grad_x = self._dx_3d(u, 1)
            grad_y = self._dy_3d(u, 1)
            grad_z = self._dz_3d(u, 1)
            return torch.stack([grad_x, grad_y, grad_z], dim=-4)

    def divergence(self, u):
        """
        Compute the divergence ∇·u for vector fields.

        Parameters
        ----------
        u : torch.Tensor
            Input vector field

        Returns
        -------
        torch.Tensor
            The divergence of the vector field
        """
        # Check input dimensions match vector field components
        n_components_expected = self.dim
        n_components_actual = u.shape[-self.dim-1]
        if n_components_actual != n_components_expected:
            raise ValueError(f"Input must be a {self.dim}D vector field with {n_components_expected} components")

        if self.dim == 1:
            return self._dx_1d(u[..., 0, :], 1)
        elif self.dim == 2:
            u1, u2 = u[..., 0, :, :], u[..., 1, :, :]
            return self._dx_2d(u1, 1) + self._dy_2d(u2, 1)
        else:  # dim == 3
            u1, u2, u3 = u[..., 0, :, :, :], u[..., 1, :, :, :], u[..., 2, :, :, :]
            return self._dx_3d(u1, 1) + self._dy_3d(u2, 1) + self._dz_3d(u3, 1)

    def curl(self, u):
        """
        Compute the curl ∇×u for vector fields.

        Parameters
        ----------
        u : torch.Tensor
            Input vector field

        Returns
        -------
        torch.Tensor
            The curl of the vector field
        """
        if self.dim == 1:
            raise ValueError("Curl is not defined for 1D")
        elif self.dim == 2:
            if u.shape[-3] != 2:
                raise ValueError("Input must be a 2D vector field with 2 components")
            u1, u2 = u[..., 0, :, :], u[..., 1, :, :]
            return self._dx_2d(u2, 1) - self._dy_2d(u1, 1)
        else:  # dim == 3
            if u.shape[-4] != 3:
                raise ValueError("Input must be a 3D vector field with 3 components") 
            u1, u2, u3 = u[..., 0, :, :, :], u[..., 1, :, :, :], u[..., 2, :, :, :]
            curl_x = self._dy_3d(u3, 1) - self._dz_3d(u2, 1)
            curl_y = self._dz_3d(u1, 1) - self._dx_3d(u3, 1)
            curl_z = self._dx_3d(u2, 1) - self._dy_3d(u1, 1)
            return torch.stack([curl_x, curl_y, curl_z], dim=-4)

    def _dx_1d(self, u, order):
        """1D derivative with respect to x."""
        if order == 1:
            return self._dx_1st_1d(u)
        elif order == 2:
            return self._dx_2nd_1d(u)
        else:
            raise ValueError(
                "Only 1st and 2nd order derivatives currently supported")
    
    def _dx_1st_1d(self, u):
        """First order derivative with respect to x (1D)."""
        
        if self.periodic_in_x:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i+1} - f_{i-1})/(2h)
            dx = (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / (2.0 * self.h[0])
            
        else:
            # Non-periodic case: handle boundaries separately
            dx = torch.zeros_like(u)
            
            # Interior points: Second-order central differences  (f_{i+1} - f_{i-1})/(2h)
            dx[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2.0 * self.h[0])
            
            # Left boundary: 3rd-order forward differences (-11f_{0} + 18f_{1} - 9f_{2} + 2f_{3})/(6h)
            dx[..., 0] = (-11 * u[..., 0] + 18 * u[..., 1] - 9 * u[..., 2] + 2 * u[..., 3]) / (6.0 * self.h[0])
            
            # Right boundary: 3rd-order backward differences (-2f_{n-4} + 9f_{n-3} - 18f_{n-2} + 11f_{n-1})/(6h)
            dx[..., -1] = (-2 * u[..., -4] + 9 * u[..., -3] - 18 * u[..., -2] + 11 * u[..., -1]) / (6.0 * self.h[0])
        
        return dx
    
    def _dx_2nd_1d(self, u):
        """Second order derivative with respect to x (1D)."""
        
        if self.periodic_in_x:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i+1} - 2f_{i} + f_{i-1})/(h²)
            dxx = (torch.roll(u, -1, dims=-1) - 2 * u + torch.roll(u, 1, dims=-1)) / (self.h[0]**2)
        
        else:
            # Non-periodic case: handle boundaries separately
            dxx = torch.zeros_like(u)
            
            # Interior points: Second-order central differences
            # (f_{i+1} - 2f_{i} + f_{i-1})/(h²)
            dxx[..., 1:-1] = (u[..., 2:] - 2 * u[..., 1:-1] + u[..., :-2]) / (self.h[0]**2)
            
            # Left boundary: 3rd-order forward differences (2f_{0} - 5f_{1} + 4f_{2} - f_{3})/h²
            dxx[..., 0] = (2 * u[..., 0] - 5 * u[..., 1] + 4 * u[..., 2] - u[..., 3]) / (self.h[0]**2)
            
            # Right boundary: 3rd-order backward differences (-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/h²
            dxx[..., -1] = (-u[..., -4] + 4 * u[..., -3] - 5 * u[..., -2] + 2 * u[..., -1]) / (self.h[0]**2)
        
        return dxx

    def _dx_2d(self, u, order):
        """2D derivative with respect to x."""
        if order == 1:
            return self._dx_1st_2d(u)
        elif order == 2:
            return self._dx_2nd_2d(u)
        else:
            raise ValueError(
                "Only 1st and 2nd order derivatives currently supported")

    def _dy_2d(self, u, order):
        """2D derivative with respect to y."""
        if order == 1:
            return self._dy_1st_2d(u)
        elif order == 2:
            return self._dy_2nd_2d(u)
        else:
            raise ValueError("Only 1st and 2nd order derivatives currently supported")
    
    def _dx_1st_2d(self, u):
        """First order derivative with respect to x (2D)."""
        
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
            dx[..., 0, :] = (-11 * u[..., 0, :] + 18 * u[..., 1, :] - 9 * u[..., 2, :] + 2 * u[..., 3, :]) / (6.0 * self.h[0])
            
            # Right boundary: 3rd-order backward differences (-2f_{n-4} + 9f_{n-3} - 18f_{n-2} + 11f_{n-1})/(6h_{x})
            dx[..., -1, :] = (-2 * u[..., -4, :] + 9 * u[..., -3, :] - 18 * u[..., -2, :] + 11 * u[..., -1, :]) / (6.0 * self.h[0])
        
        return dx
    
    def _dy_1st_2d(self, u):
        """First order derivative with respect to y (2D)."""
        
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
            dy[..., :, 0] = (-11 * u[..., :, 0] + 18 * u[..., :, 1] - 9 * u[..., :, 2] + 2 * u[..., :, 3]) / (6.0 * self.h[1])

            # Top boundary: 3rd-order backward differences (-2f_{n-4} + 9f_{n-3} - 18f_{n-2} + 11f_{n-1})/(6h_{y})
            dy[..., :, -1] = (-2 * u[..., :, -4] + 9 * u[..., :, -3] - 18 * u[..., :, -2] + 11 * u[..., :, -1]) / (6.0 * self.h[1])
        
        return dy
    
    def _dx_2nd_2d(self, u):
        """Second order derivative with respect to x (2D)."""
        
        if self.periodic_in_x:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i+1,j} - 2f_{i,j} + f_{i-1,j})/(h_{x}²)
            dxx = (torch.roll(u, -1, dims=-2) - 2 * u + torch.roll(u, 1, dims=-2)) / (self.h[0] ** 2)
        
        else:
            # Non-periodic case: handle boundaries separately
            dxx = torch.zeros_like(u)
            
            # Interior points: Second-order central differences
            # (f_{i+1,j} - 2f_{i,j} + f_{i-1,j})/(h_{x}²)
            dxx[..., 1:-1, :] = (u[..., 2:, :] - 2 * u[..., 1:-1, :] + u[..., :-2, :]) / (self.h[0] ** 2)
            
            # Left boundary: 3rd-order forward differences (2f_{0} - 5f_{1} + 4f_{2} - f_{3})/h_{x}²
            dxx[..., 0, :] = (2 * u[..., 0, :] - 5 * u[..., 1, :] + 4 * u[..., 2, :] - u[..., 3, :]) / (self.h[0] ** 2)
            
            # Right boundary: 3rd-order backward differences (-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/h_{x}²
            dxx[..., -1, :] = (-u[..., -4, :] + 4 * u[..., -3, :] - 5 * u[..., -2, :] + 2 * u[..., -1, :]) / (self.h[0] ** 2)
        
        return dxx
    
    def _dy_2nd_2d(self, u):
        """Second order derivative with respect to y (2D)."""
        
        if self.periodic_in_y:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i,j+1} - 2f_{i,j} + f_{i,j-1})/(h_{y}²)
            dyy = (torch.roll(u, -1, dims=-1) - 2 * u + torch.roll(u, 1, dims=-1)) / (self.h[1] ** 2)
        
        else:
            # Non-periodic case: handle boundaries separately
            dyy = torch.zeros_like(u)
            
            # Interior points: Second-order central differences
            # (f_{i,j+1} - 2f_{i,j} + f_{i,j-1})/(h_{y}²)
            dyy[..., :, 1:-1] = (u[..., :, 2:] - 2 * u[..., :, 1:-1] + u[..., :, :-2]) / (self.h[1] ** 2)
            
            # Bottom boundary: 3rd-order forward differences (2f_{0} - 5f_{1} + 4f_{2} - f_{3})/h_{y}²
            dyy[..., :, 0] = (2 * u[..., :, 0] - 5 * u[..., :, 1] + 4 * u[..., :, 2] - u[..., :, 3]) / (self.h[1] ** 2)
            
            # Top boundary: 3rd-order backward differences (-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/h_{y}²
            dyy[..., :, -1] = (-u[..., :, -4] + 4 * u[..., :, -3] - 5 * u[..., :, -2] + 2 * u[..., :, -1]) / (self.h[1] ** 2)
        
        return dyy
    
    def _dx_3d(self, u, order):
        """3D derivative with respect to x."""
        if order == 1:
            return self._dx_1st_3d(u)
        elif order == 2:
            return self._dx_2nd_3d(u)
        else:
            raise ValueError("Only 1st and 2nd order derivatives currently supported")

    def _dy_3d(self, u, order):
        """3D derivative with respect to y."""
        if order == 1:
            return self._dy_1st_3d(u)
        elif order == 2:
            return self._dy_2nd_3d(u)
        else:
            raise ValueError("Only 1st and 2nd order derivatives currently supported")

    def _dz_3d(self, u, order):
        """3D derivative with respect to z."""
        if order == 1:
            return self._dz_1st_3d(u)
        elif order == 2:
            return self._dz_2nd_3d(u)
        else:
            raise ValueError(
                "Only 1st and 2nd order derivatives currently supported")
    
    def _dx_1st_3d(self, u):
        """First order derivative with respect to x (3D)."""
        
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
            dx[..., 0, :, :] = (-11 * u[..., 0, :, :] + 18 * u[..., 1, :, :] - 9 * u[..., 2, :, :] + 2 * u[..., 3, :, :]) / (6.0 * self.h[0])

            # Right boundary: 3rd-order backward differences (-2f_{n-4} + 9f_{n-3} - 18f_{n-2} + 11f_{n-1})/(6h_{x})
            dx[..., -1, :, :] = (-2 * u[..., -4, :, :] + 9 * u[..., -3, :, :] - 18 * u[..., -2, :, :] + 11 * u[..., -1, :, :]) / (6.0 * self.h[0])
        
        return dx
    
    def _dy_1st_3d(self, u):
        """First order derivative with respect to y (3D)."""
        
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
            dy[..., :, 0, :] = (-11 * u[..., :, 0, :] + 18 * u[..., :, 1, :] - 9 * u[..., :, 2, :] + 2 * u[..., :, 3, :]) / (6.0 * self.h[1])

            # Top boundary: 3rd-order backward differences (-2f_{n-4} + 9f_{n-3} - 18f_{n-2} + 11f_{n-1})/(6h_{y})
            dy[..., :, -1, :] = (-2 * u[..., :, -4, :] + 9 * u[..., :, -3, :] - 18 * u[..., :, -2, :] + 11 * u[..., :, -1, :]) / (6.0 * self.h[1])
        
        return dy
    
    def _dz_1st_3d(self, u):
        """First order derivative with respect to z (3D)."""
        
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
            dz[..., :, :, 0] = (-11 * u[..., :, :, 0] + 18 * u[..., :, :, 1] - 9 * u[..., :, :, 2] + 2 * u[..., :, :, 3]) / (6.0 * self.h[2])

            # Back boundary: 3rd-order backward differences (-2f_{n-4} + 9f_{n-3} - 18f_{n-2} + 11f_{n-1})/(6h_{z})
            dz[..., :, :, -1] = (-2 * u[..., :, :, -4] + 9 * u[..., :, :, -3] - 18 * u[..., :, :, -2] + 11 * u[..., :, :, -1]) / (6.0 * self.h[2])
        
        return dz
    
    def _dx_2nd_3d(self, u):
        """Second order derivative with respect to x (3D)."""
        
        if self.periodic_in_x:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i+1,j,k} - 2f_{i,j,k} + f_{i-1,j,k})/(h_{x}²)
            dxx = (torch.roll(u, -1, dims=-3) - 2 * u + torch.roll(u, 1, dims=-3)) / (self.h[0] ** 2)
        
        else:
            # Non-periodic case: handle boundaries separately
            dxx = torch.zeros_like(u)
            
            # Interior points: Second-order central differences
            # (f_{i+1,j,k} - 2f_{i,j,k} + f_{i-1,j,k})/(h_{x}²)
            dxx[..., 1:-1, :, :] = (u[..., 2:, :, :] - 2 * u[..., 1:-1, :, :] + u[..., :-2, :, :]) / (self.h[0] ** 2)
            
            # Left boundary: 3rd-order forward differences (2f_{0} - 5f_{1} + 4f_{2} - f_{3})/h_{x}²
            dxx[..., 0, :, :] = (2 * u[..., 0, :, :] - 5 * u[..., 1, :, :] + 4 * u[..., 2, :, :] - u[..., 3, :, :]) / (self.h[0] ** 2)
            
            # Right boundary: 3rd-order backward differences (-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/h_{x}²
            dxx[..., -1, :, :] = (-u[..., -4, :, :] + 4 * u[..., -3, :, :] - 5 * u[..., -2, :, :] + 2 * u[..., -1, :, :]) / (self.h[0] ** 2)
        
        return dxx
    
    def _dy_2nd_3d(self, u):
        """Second order derivative with respect to y (3D)."""
        
        if self.periodic_in_y:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i,j+1,k} - 2f_{i,j,k} + f_{i,j-1,k})/(h_{y}²)
            dyy = (torch.roll(u, -1, dims=-2) - 2 * u + torch.roll(u, 1, dims=-2)) / (self.h[1] ** 2)
        
        else:
            # Non-periodic case: handle boundaries separately
            dyy = torch.zeros_like(u)
            
            # Interior points: Second-order central differences
            # (f_{i,j+1,k} - 2f_{i,j,k} + f_{i,j-1,k})/(h_{y}²)
            dyy[..., :, 1:-1, :] = (u[..., :, 2:, :] - 2 * u[..., :, 1:-1, :] + u[..., :, :-2, :]) / (self.h[1] ** 2)
            
            # Bottom boundary: 3rd-order forward differences (2f_{0} - 5f_{1} + 4f_{2} - f_{3})/h_{y}²
            dyy[..., :, 0, :] = (2 * u[..., :, 0, :] - 5 * u[..., :, 1, :] + 4 * u[..., :, 2, :] - u[..., :, 3, :]) / (self.h[1] ** 2)
            
            # Top boundary: 3rd-order backward differences (-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/h_{y}²
            dyy[..., :, -1, :] = (-u[..., :, -4, :] + 4 * u[..., :, -3, :] - 5 * u[..., :, -2, :] + 2 * u[..., :, -1, :]) / (self.h[1] ** 2)
        
        return dyy
    
    def _dz_2nd_3d(self, u):
        """Second order derivative with respect to z (3D)."""
        
        if self.periodic_in_z:
            # Periodic case: use torch.roll for boundary wrapping
            # Central difference: (f_{i,j,k+1} - 2f_{i,j,k} + f_{i,j,k-1})/(h_{z}²)
            dzz = (torch.roll(u, -1, dims=-1) - 2 * u +
                   torch.roll(u, 1, dims=-1)) / (self.h[2] ** 2)
        
        else:
            # Non-periodic case: handle boundaries separately
            dzz = torch.zeros_like(u)
            
            # Interior points: Second-order central differences
            # (f_{i,j,k+1} - 2f_{i,j,k} + f_{i,j,k-1})/(h_{z}²)
            dzz[..., :, :, 1:-1] = (u[..., :, :, 2:] - 2 * u[..., :, :, 1:-1] + u[..., :, :, :-2]) / (self.h[2] ** 2)
            
            # Front boundary: 3rd-order forward differences (2f_{0} - 5f_{1} + 4f_{2} - f_{3})/h_{z}²
            dzz[..., :, :, 0] = (2 * u[..., :, :, 0] - 5 * u[..., :, :, 1] + 4 * u[..., :, :, 2] - u[..., :, :, 3]) / (self.h[2] ** 2)
            
            # Back boundary: 3rd-order backward differences (-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/h_{z}²
            dzz[..., :, :, -1] = (-u[..., :, :, -4] + 4 * u[..., :, :, -3] - 5 * u[..., :, :, -2] + 2 * u[..., :, :, -1]) / (self.h[2] ** 2)
        
        return dzz
    

# Backward compatibility functions
def central_diff_1d(x, h, periodic_in_x=True):
    """
    Backward compatibility function for central_diff_1d.
    Creates a FiniteDiff instance with dim=1 and returns dx.
    """
    fd1d = FiniteDiff(dim=1, h=h, periodic_in_x=periodic_in_x)
    return fd1d.dx(x)


def central_diff_2d(x, h, periodic_in_x=True, periodic_in_y=True):
    """
    Backward compatibility function for central_diff_2d.
    Creates a FiniteDiff instance with dim=2 and returns dx, dy.
    """
    fd2d = FiniteDiff(
        dim=2, h=h, periodic_in_x=periodic_in_x, periodic_in_y=periodic_in_y
    )
    return fd2d.dx(x), fd2d.dy(x)


def central_diff_3d(
        x,
        h,
        periodic_in_x=True,
        periodic_in_y=True,
        periodic_in_z=True):
    """
    Backward compatibility function for central_diff_3d.
    Creates a FiniteDiff instance with dim=3 and returns dx, dy, dz.
    """
    fd3d = FiniteDiff(
        dim=3,
        h=h,
        periodic_in_x=periodic_in_x,
        periodic_in_y=periodic_in_y,
        periodic_in_z=periodic_in_z,
    )
    return fd3d.dx(x), fd3d.dy(x), fd3d.dz(x)


def get_non_uniform_fd_weights(
        points,
        num_neighbors=5,
        derivative_indices=[0],
        radius=None,
        regularize_lstsq=False):
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
    distances, indices = torch.topk(
        pairwise_distances, k=k, dim=1, largest=False)
    
    # Get mask for neighbors within cutoff radius (and always keep at least 3)
    if radius is None:
        radius_mask = torch.ones_like(distances, dtype=torch.bool)
    else:
        radius_mask = distances <= radius
        radius_mask[:, :3] = True

    # Initialize A to 1 since first row for each point and derivative is 1
    A = torch.ones((N, d + 1, k), dtype=points.dtype, device=points.device)
    # Compute coordinate differences
    for i in range(d):
        A[:, i + 1, :] = points[indices, i] - points[:, i].unsqueeze(1)
    # Repeat it for each derivative to be evaluated so it becomes of shape (N,
    # len(derivative_indices), d+1, k)
    A = A.unsqueeze(1).expand(-1, len(derivative_indices), -1, -1)

    # Zero out columns for neighbors that are not within the radius
    A = A * radius_mask.unsqueeze(1).unsqueeze(2)
    
    # Compute right hand side
    b = torch.zeros((len(derivative_indices), d + 1, 1),
                    dtype=points.dtype, device=points.device)
    for i in range(len(derivative_indices)):
        b[i, derivative_indices[i] + 1] = 1
    # Repeat so it becomes (N, len(derivative_indices), d+1, 1)
    b = b.unsqueeze(0).expand(N, -1, -1, -1)

    # Solve least squares system Aw = b  
    #    sometimes torch.linalg.lstsq(A, b).solution creates artifacts so can add regularizer
    # but regularizer can deteriorate performance when system is
    # well-conditioned

    if regularize_lstsq:
        lambda_reg = 1e-6
        I_k = torch.eye(
            k,
            dtype=A.dtype,
            device=A.device).unsqueeze(0).unsqueeze(0)

        AT = A.transpose(-2, -1)
        AT_b = torch.matmul(AT, b)
        AT_A = torch.matmul(AT, A) + lambda_reg * I_k

        # Use Cholesky decomposition to accelerate torch.linalg.solve(AT_A,
        # AT_b).squeeze(-1)
        fd_weights = torch.cholesky_solve(
            AT_b, torch.linalg.cholesky(AT_A)).squeeze(-1)

    else:
        fd_weights = torch.linalg.lstsq(A, b).solution 

    return indices, fd_weights.squeeze(-1)


def non_uniform_fd(
    points,
    values,
    num_neighbors=5,
    derivative_indices=[0],
    radius=None,
    regularize_lstsq=False,
):
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

    indices, fd_weights = get_non_uniform_fd_weights(
        points=points,
                                                    num_neighbors=num_neighbors, 
                                                    derivative_indices=derivative_indices,
                                                    radius=radius,
        regularize_lstsq=regularize_lstsq,
    )

    derivatives = torch.einsum("nij,nj->in", fd_weights, values[indices])

    return derivatives


class FourierDiff:
    """
    A unified class for computing Fourier/spectral derivatives in 1D, 2D, 3D.

    This class provides comprehensive methods for computing derivatives using Fourier/spectral
    methods with support for both periodic and non-periodic functions through Fourier continuation.

    Fourier Spectral Methods
    ------------------------
    The class implements spectral differentiation for
    - Periodic functions: Direct Fourier differentiation using FFT 
    - Non-periodic functions: Fourier continuation (FC) is used to extend functions 
          to larger domain on which the functions are periodic before applying 
          Fourier differentiation with FFT.

    Also provides gradient, divergence, curl, and Laplacian operations

    Mathematical Background
    -----------------------
    For periodic functions on [0, 2π], the derivative is computed as:
    - Forward transform: û_k = FFT(u)
    - Derivative in Fourier space: (∂u/∂x)^_k = ik * û_k
    - Inverse transform: ∂u/∂x = IFFT(ik * û_k)

    For non-periodic functions, Fourier continuation extends the function to 
    an extended domain (e.g. [0, 2π] → [0, 2π + 2π*additional_pts/n]) on which
    the function is periodic.
    

    Available Methods
    ----------------
    Derivative Methods:
    - dx(u, order=1): Compute derivative with respect to x
    - dy(u, order=1): Compute derivative with respect to y (2D/3D only)
    - dz(u, order=1): Compute derivative with respect to z (3D only)
    - derivative(u, order): Compute derivative with order tuple (e.g., (1,0) for ∂/∂x)

    Vector Calculus Operators:
    - laplacian(u): Compute the Laplacian ∇²f
    - gradient(u): Compute the gradient ∇f (returns vector field)
    - divergence(u): Compute the divergence ∇·u (for vector fields)
    - curl(u): Compute the curl ∇×u (for vector fields, 2D/3D only)

    Examples
    --------
    >>> # 1D Fourier derivatives
    >>> x = torch.linspace(0, 2*torch.pi, 100)
    >>> u = torch.sin(x)
    >>> fd1d = FourierDiff(dim=1, L=2*torch.pi, use_fc=False)
    >>> du_dx = fd1d.dx(u)  # First derivative
    >>> d2u_dx2 = fd1d.dx(u, order=2)  # Second derivative
    >>>
    >>> # 2D Fourier derivatives
    >>> fd2d = FourierDiff(dim=2, L=(2*torch.pi, 2*torch.pi), use_fc=False)
    >>> x = torch.linspace(0, 2*torch.pi, 50)
    >>> y = torch.linspace(0, 2*torch.pi, 50)
    >>> X, Y = torch.meshgrid(x, y, indexing='ij')
    >>> u = torch.sin(X) * torch.cos(Y)
    >>> du_dx = fd2d.dx(u)
    >>> du_dy = fd2d.dy(u)
    >>> grad = fd2d.gradient(u)  # Returns [du_dx, du_dy]
    >>>
    >>> # 3D Fourier derivatives
    >>> fd3d = FourierDiff(dim=3, L=(2*torch.pi, 2*torch.pi, 2*torch.pi), use_fc=False)
    >>> x = torch.linspace(0, 2*torch.pi, 20)
    >>> y = torch.linspace(0, 2*torch.pi, 20)
    >>> z = torch.linspace(0, 2*torch.pi, 20)
    >>> X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    >>> u = torch.sin(X) * torch.cos(Y) * torch.sin(Z)  # 3D scalar field
    >>> du_dx = fd3d.dx(u)
    >>> du_dy = fd3d.dy(u)
    >>> du_dz = fd3d.dz(u)
    >>> laplacian = fd3d.laplacian(u)  
    >>>
    >>> # Vector field operations
    >>> vx = torch.sin(X) * torch.cos(Y) * torch.sin(Z)
    >>> vy = torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    >>> vz = torch.sin(X) * torch.sin(Y) * torch.cos(Z)
    >>> v = torch.stack([vx, vy, vz], dim=-4) 
    >>> div_v = fd3d.divergence(v) 
    >>> curl_v = fd3d.curl(v) 

    """

    def __init__(
        self,
        dim,
        L=None,
        use_fc=False,
        fc_degree=4,
        fc_n_additional_pts=50,
        low_pass_filter_ratio=None,
    ):
        """
        Initialize the FourierDiff class for computing Fourier derivatives.
        
        Parameters
        ----------
        dim : int
            Dimension of the input field. Must be 1, 2, or 3.
        L : float or tuple, optional
            Length of the domain for Fourier differentiation. By default 2*pi for each dimension.
        use_fc : str, optional
            Whether to use Fourier continuation for non-periodic functions.
            Options: False (no FC), 'Legendre', 'Gram'. By default False.
        fc_degree : int, optional
            Degree of the Fourier continuation polynomial matching. This is the number
            of matching points on the left and right boundaries used for the Fourier
            continuation procedure. By default 4.
        fc_n_additional_pts : int, optional
            Number of additional points to add with the Fourier continuation layer.
            This extends the domain to handle non-periodic functions. By default 50.
        low_pass_filter_ratio : float, optional
            If not None, apply a low-pass filter to the Fourier coefficients to reduce
            high-frequency noise. Should be between 0 and 1. By default None.

        """

        # Check if dim is valid
        if dim not in [1, 2, 3]:
            raise ValueError("dim must be 1, 2, or 3")

        self.dim = dim
        
        # Set default L based on dimension
        if L is None:
            L = 2 * torch.pi
        if not isinstance(L, (tuple, list)):
            L = (L,) * dim
        if len(L) != dim:
            raise ValueError(f"For {dim}D, L must be a single float or tuple with {dim} elements")
        self.L = L[0] if dim == 1 else L

        self.use_fc = use_fc
        self.fc_degree = fc_degree
        self.fc_n_additional_pts = fc_n_additional_pts
        self.low_pass_filter_ratio = low_pass_filter_ratio

        # Initialize FC class if needed
        self.FC = None
        if self.use_fc in ['Legendre', 'Gram']:
            FC_class = FCLegendre if self.use_fc == 'Legendre' else FCGram
            self.FC = FC_class(d=self.fc_degree, n_additional_pts=self.fc_n_additional_pts)
    
    def compute_multiple_derivatives(self, u, derivatives):
        """
        Compute multiple derivatives in a single FFT/IFFT call for better performance.
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor.
        derivatives : list
            List of derivative specifications:
            - 1D: list of int (orders)
            - 2D: list of tuples (order_x, order_y)
            - 3D: list of tuples (order_x, order_y, order_z)
            
        Returns
        -------
        list of torch.Tensor
            List of computed derivatives in the same order as derivatives input
        """
        if self.dim == 1:
            return self._compute_multiple_derivatives_1d(u, derivatives)
        elif self.dim == 2:
            return self._compute_multiple_derivatives_2d(u, derivatives)
        elif self.dim == 3:
            return self._compute_multiple_derivatives_3d(u, derivatives)

    def derivative(self, u, order):
        """
        Compute Fourier derivative of a given tensor.
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor
        order : tuple
            Derivative orders:
            - 1D: (order_x,)
            - 2D: (order_x, order_y)
            - 3D: (order_x, order_y, order_z)
            
        Returns
        -------
        torch.Tensor
            The derivative of the input tensor
        """
        if len(order) != self.dim:
            raise ValueError(f"For {self.dim}D, order must be a tuple with {self.dim} elements")
            
        if self.dim == 1:
            derivatives = self._compute_multiple_derivatives_1d(u, [order[0]])
        elif self.dim == 2:
            derivatives = self._compute_multiple_derivatives_2d(u, [order])
        elif self.dim == 3:
            derivatives = self._compute_multiple_derivatives_3d(u, [order])
        
        return derivatives[0]

    def partial(self, u, direction="x", order=1):
        """
        Compute partial Fourier derivative along a specific direction.
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor
        direction : str, optional
            Direction along which to compute the derivative, by default 'x'
            Options: 'x', 'y' (2D/3D only), 'z' (3D only)
        order : int, optional
            Order of the derivative, by default 1
            
        Returns
        -------
        torch.Tensor
            The partial derivative of the input tensor
        """
        if direction == "x":
            return self.dx(u, order=order)
        elif direction == "y" and self.dim >= 2:
            return self.dy(u, order=order)
        elif direction == "z" and self.dim >= 3:
            return self.dz(u, order=order)
        else:
            raise ValueError(f"Invalid direction '{direction}' for dimension {self.dim}")
    
    def dx(self, u, order=1):
        """Compute derivative with respect to x."""
        if self.dim == 1:
            return self._dx_1d(u, order)
        elif self.dim == 2:
            return self._dx_2d(u, order)
        elif self.dim == 3:
            return self._dx_3d(u, order)
    
    def dy(self, u, order=1):
        """Compute derivative with respect to y (2D/3D only)."""
        if self.dim < 2:
            raise ValueError("dy method only available for 2D and 3D")
        elif self.dim == 2:
            return self._dy_2d(u, order)
        elif self.dim == 3:
            return self._dy_3d(u, order)

    def dz(self, u, order=1):
        """Compute derivative with respect to z (3D only)."""
        if self.dim < 3:
            raise ValueError("dz method only available for 3D")
        return self._dz_3d(u, order)
    
    def laplacian(self, u):
        """Compute the Laplacian ∇²f."""
        if self.dim == 1:
            return self.dx(u, order=2)
        elif self.dim == 2:
            return self.dx(u, order=2) + self.dy(u, order=2)
        elif self.dim == 3:
            return self.dx(u, order=2) + self.dy(u, order=2) + self.dz(u, order=2)

    def gradient(self, u):
        """Compute the gradient ∇f (returns vector field)."""
        if self.dim == 1:
            return self.dx(u).unsqueeze(-2)
        elif self.dim == 2:
            return torch.stack([self.dx(u), self.dy(u)], dim=-3)
        elif self.dim == 3:
            return torch.stack([self.dx(u), self.dy(u), self.dz(u)], dim=-4)
    
    def divergence(self, u):
        """Compute the divergence ∇·u (for vector fields)."""
        expected_dims = {1: 1, 2: 2, 3: 3}
        if u.shape[-self.dim-1] != expected_dims[self.dim]:
            raise ValueError(
                f"For {self.dim}D, input must have {expected_dims[self.dim]} components in the vector dimension"
            )

        if self.dim == 1:
            return self.dx(u.squeeze(-2))
        elif self.dim == 2:
            return self.dx(u[..., 0, :, :]) + self.dy(u[..., 1, :, :])
        elif self.dim == 3:
            return self.dx(u[..., 0, :, :, :]) + self.dy(u[..., 1, :, :, :]) + self.dz(u[..., 2, :, :, :])
    
    def curl(self, u):
        """Compute the curl ∇×u (for vector fields, 2D/3D only)."""
        # Check input dimensions
        if self.dim == 1:
            raise ValueError("curl not defined for 1D")
        elif self.dim == 2 and u.shape[-3] != 2:
            raise ValueError("For 2D, input must have 2 components in the vector dimension")
        elif self.dim == 3 and u.shape[-4] != 3:
            raise ValueError("For 3D, input must have 3 components in the vector dimension")
        if self.dim == 2:
            # In 2D: ∂v/∂x - ∂u/∂y where u = (u,v) is a 2D vector field
            return self.dx(u[..., 1, :, :]) - self.dy(u[..., 0, :, :])
        elif self.dim == 3:
            # In 3D, ∇×u = (∂w/∂y - ∂v/∂z, ∂u/∂z - ∂w/∂x, ∂v/∂x - ∂u/∂y) where u = (u,v,w) is a 3D vector field
            curl_x = self.dy(u[..., 2, :, :, :]) - self.dz(u[..., 1, :, :, :])  # ∂w/∂y - ∂v/∂z
            curl_y = self.dz(u[..., 0, :, :, :]) - self.dx(u[..., 2, :, :, :])  # ∂u/∂z - ∂w/∂x  
            curl_z = self.dx(u[..., 1, :, :, :]) - self.dy(u[..., 0, :, :, :])  # ∂v/∂x - ∂u/∂y
            
            # Stack the components into a 3D vector field
            return torch.stack([curl_x, curl_y, curl_z], dim=-4)

    def _compute_multiple_derivatives_1d(self, u, orders):
        """1D multiple derivatives computation."""
        if u is None:
            raise ValueError("Input tensor u is None")

        L_x = self.L
        nx = u.shape[-1]
        u_clone = u.clone()

        # Apply Fourier continuation if specified
        if self.use_fc and self.FC is not None:
            FC = self.FC.to(u_clone.device)
            u_clone = FC(u_clone, dim=1)
            L_x *= (nx + self.fc_n_additional_pts) / nx

        # Update grid parameters after extension
        nx = u_clone.shape[-1]
        dx = L_x / nx

        # FFT
        u_h = torch.fft.rfft(u_clone, dim=-1)

        # Frequency array
        k_x = torch.fft.rfftfreq(nx, d=dx, device=u_h.device) * (2 * torch.pi)

        # Apply low-pass filter if specified
        if self.low_pass_filter_ratio is not None:
            cutoff = int(u_h.shape[-1] * self.low_pass_filter_ratio)
            u_h[..., cutoff:] = 0

        # Compute derivatives
        results = []
        for order in orders:
            derivative_u_h = ((1j * k_x) ** order) * u_h
            results.append(derivative_u_h)

        derivatives_ft = torch.stack(results, dim=0)
        derivatives_real = torch.fft.irfft(derivatives_ft, dim=-1, n=nx)

        # Crop result if Fourier continuation was used
        if self.use_fc and self.FC is not None:
            FC = self.FC.to(derivatives_real.device)
            derivatives_real = FC.restrict(derivatives_real, dim=1)

        return [derivatives_real[i] for i in range(len(orders))]
    
    def _dx_1d(self, u, order):
        """1D derivative with respect to x."""
        derivatives = self._compute_multiple_derivatives_1d(u, [order])
        return derivatives[0]
    
    def _compute_multiple_derivatives_2d(self, u, derivatives):
        """2D multiple derivatives computation."""
        if u is None:
            raise ValueError("Input tensor u is None")

        L_x, L_y = self.L[0], self.L[1]
        nx, ny = u.shape[-2], u.shape[-1]
        u_clone = u.clone()

        # Apply Fourier continuation if specified
        if self.use_fc and self.FC is not None:
            FC = self.FC.to(u_clone.device)
            u_clone = FC.extend2d(u_clone)
            L_x *= (nx + self.fc_n_additional_pts) / nx
            L_y *= (ny + self.fc_n_additional_pts) / ny

        # Update grid parameters after extension
        nx, ny = u_clone.shape[-2], u_clone.shape[-1]
        dx, dy = L_x / nx, L_y / ny

        # FFT with transposed axes (shape -> (ny, nx))
        u_h = torch.fft.fft2(u_clone.transpose(-2, -1), dim=(-2, -1))

        # Frequency arrays
        k_x = torch.fft.fftfreq(nx, d=dx, device=u_h.device) * (2 * torch.pi)
        k_y = torch.fft.fftfreq(ny, d=dy, device=u_h.device) * (2 * torch.pi)

        # Create frequency meshgrid
        KY, KX = torch.meshgrid(k_y, k_x, indexing="ij")  

        # Apply low-pass filter if specified
        if self.low_pass_filter_ratio is not None:
            cutoff_x = int(nx * self.low_pass_filter_ratio)
            cutoff_y = int(ny * self.low_pass_filter_ratio)
            u_h[..., cutoff_y:, :] = 0
            u_h[..., :, cutoff_x:] = 0

        # Compute derivatives
        results = []
        for order_x, order_y in derivatives:
            # Expand meshgrid tensors for proper broadcasting
            KX_expanded = KX.expand(u_h.shape)
            KY_expanded = KY.expand(u_h.shape)
            
            derivative_u_h = ((1j * KX_expanded) ** order_x) * ((1j * KY_expanded) ** order_y) * u_h
            results.append(derivative_u_h)

        derivatives_ft = torch.stack(results, dim=0)
        derivatives_real = torch.fft.ifft2(derivatives_ft, dim=(-2, -1)).real

        # Transpose back to original shape (nx, ny)
        derivatives_real = derivatives_real.transpose(-2, -1)

        # Crop result if Fourier continuation was used
        if self.use_fc and self.FC is not None:
            FC = self.FC.to(derivatives_real.device)
            derivatives_real = FC.restrict(derivatives_real, dim=2)

        return [derivatives_real[i] for i in range(len(derivatives))]
    
    def _dx_2d(self, u, order):
        """2D derivative with respect to x."""
        derivatives = self._compute_multiple_derivatives_2d(u, [(order, 0)])
        return derivatives[0]
    
    def _dy_2d(self, u, order):
        """2D derivative with respect to y."""
        derivatives = self._compute_multiple_derivatives_2d(u, [(0, order)])
        return derivatives[0]
    
    def _compute_multiple_derivatives_3d(self, u, derivatives):
        """3D multiple derivatives computation."""
        if u is None:
            raise ValueError("Input tensor u is None")

        L_x, L_y, L_z = self.L[0], self.L[1], self.L[2]
        nx, ny, nz = u.shape[-3], u.shape[-2], u.shape[-1]
        u_clone = u.clone()

        # Apply Fourier continuation if specified
        if self.use_fc and self.FC is not None:
            FC = self.FC.to(u_clone.device)
            u_clone = FC.extend3d(u_clone)
            L_x *= (nx + self.fc_n_additional_pts) / nx
            L_y *= (ny + self.fc_n_additional_pts) / ny
            L_z *= (nz + self.fc_n_additional_pts) / nz

        # Update grid parameters after extension
        nx, ny, nz = u_clone.shape[-3], u_clone.shape[-2], u_clone.shape[-1]
        dx, dy, dz = L_x / nx, L_y / ny, L_z / nz

        # FFT with permuted axes (shape -> (nz, ny, nx))
        u_clone_permuted = u_clone.permute(*range(u_clone.ndim-3), -1, -2, -3)
        u_h = torch.fft.fftn(u_clone_permuted, dim=(-3, -2, -1))

        # Frequency arrays
        k_x = torch.fft.fftfreq(nx, d=dx, device=u_h.device) * (2 * torch.pi)
        k_y = torch.fft.fftfreq(ny, d=dy, device=u_h.device) * (2 * torch.pi)
        k_z = torch.fft.fftfreq(nz, d=dz, device=u_h.device) * (2 * torch.pi)

        # Create frequency meshgrid
        KZ, KY, KX = torch.meshgrid(k_z, k_y, k_x, indexing="ij")
   
        # Apply low-pass filter if specified
        if self.low_pass_filter_ratio is not None:
            cutoff_x = int(nx * self.low_pass_filter_ratio)
            cutoff_y = int(ny * self.low_pass_filter_ratio)
            cutoff_z = int(nz * self.low_pass_filter_ratio)
            u_h[..., cutoff_y:, :, :] = 0
            u_h[..., :, cutoff_x:, :] = 0
            u_h[..., :, :, cutoff_z:] = 0

        # Compute derivatives
        results = []
        for order_x, order_y, order_z in derivatives:
            # Expand meshgrid tensors for proper broadcasting
            KX_expanded = KX.expand(u_h.shape)
            KY_expanded = KY.expand(u_h.shape)
            KZ_expanded = KZ.expand(u_h.shape)
            
            derivative_u_h = ((1j * KX_expanded) ** order_x) * ((1j * KY_expanded) ** order_y) * ((1j * KZ_expanded) ** order_z) * u_h
            results.append(derivative_u_h)

        derivatives_ft = torch.stack(results, dim=0)
        derivatives_real = torch.fft.ifftn(derivatives_ft, dim=(-3, -2, -1)).real

        # Permute back to original shape (..., nx, ny, nz)
        derivatives_real = derivatives_real.permute(*range(derivatives_real.ndim-3), -1, -2, -3)

        # Crop result if Fourier continuation was used
        if self.use_fc and self.FC is not None:
            FC = self.FC.to(derivatives_real.device)
            derivatives_real = FC.restrict(derivatives_real, dim=3)

        return [derivatives_real[i] for i in range(len(derivatives))]
       
    def _dx_3d(self, u, order):
        """3D derivative with respect to x."""
        derivatives = self._compute_multiple_derivatives_3d(u, [(order, 0, 0)])
        return derivatives[0]

    def _dy_3d(self, u, order):
        """3D derivative with respect to y."""
        derivatives = self._compute_multiple_derivatives_3d(u, [(0, order, 0)])
        return derivatives[0]
    
    def _dz_3d(self, u, order):
        """3D derivative with respect to z."""
        derivatives = self._compute_multiple_derivatives_3d(u, [(0, 0, order)])
        return derivatives[0]
