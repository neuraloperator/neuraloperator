from ..layers.fourier_continuation import FCLegendre, FCGram
import torch


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


# class FourierDiff1D:
#    """
#    A class for computing 1D Fourier derivatives with Fourier continuation support.
#
#    Provides methods for computing 1D Fourier derivatives,
#    with optional Fourier continuation for handling non-periodic functions.
#    """
#
#    def __init__(self,
#                 L=2*torch.pi,
#                 use_fc=False, 
#                 FC_d=4,
#                 FC_n_additional_pts=50,
#                 low_pass_filter_ratio=None):
#        """
#        Parameters
#        ----------
#        L : float, optional
#            Length of the domain, by default 2*pi
#        use_fc : str, optional
#            Whether to use Fourier continuation. Use for non-periodic functions.
#            Options: None, 'Legendre', 'Gram', by default False
#        FC_d : int, optional
#            'Degree' of the Fourier continuation, by default 4
#        FC_n_additional_pts : int, optional
#            Number of points to add using the Fourier continuation layer, by default 50
#        low_pass_filter_ratio : float, optional
#            If not None, apply a low-pass filter to the Fourier coefficients, by default None
#        """
#        self.L = L
#        self.use_fc = use_fc
#        self.FC_d = FC_d
#        self.fc_n_additional_pts = FC_n_additional_pts
#        self.low_pass_filter_ratio = low_pass_filter_ratio
#
#
#    def compute_multiple_derivatives(self, u, orders):
#        """
#        Compute multiple derivatives in a single FFT/IFFT call for better performance.
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input tensor. Expected shape: (..., length)
#        orders : list of int
#            List of derivative orders to compute.
#            Example: [1, 2, 3] for first, second, and third derivatives
#
#        Returns
#        -------
#        list of torch.Tensor
#            List of computed derivatives in the same order as the input orders list
#        """
#        if u is None:
#            raise ValueError("Input tensor u is None")
#
#        L_x = self.L
#        nx = u.shape[-1]
#        u_clone = u.clone()
#
#        # Apply Fourier continuation if specified
#        if self.use_fc == 'Legendre':
#            FC = FCLegendre(d=self.fc_degree, n_additional_pts=self.fc_n_additional_pts).to(u_clone.device)
#            u_clone = FC(u_clone, dim=1)
#            L_x *= (nx + self.fc_n_additional_pts) / nx
#        elif self.use_fc == 'Gram':
#            FC = FCGram(d=self.fc_degree, n_additional_pts=self.fc_n_additional_pts).to(u_clone.device)
#            u_clone = FC(u_clone, dim=1)
#            L_x *= (nx + self.fc_n_additional_pts) / nx
#
#        # Update grid parameters after extension
#        nx = u_clone.shape[-1]
#        dx = L_x / nx
#
#        # FFT
#        u_h = torch.fft.rfft(u_clone, dim=-1)
#
#        # Frequency array
#        k_x = torch.fft.rfftfreq(nx, d=dx, device=u_h.device) * (2 * torch.pi)
#
#        # Apply low-pass filter if specified
#        if self.low_pass_filter_ratio is not None:
#            cutoff = int(u_h.shape[-1] * self.low_pass_filter_ratio)
#            u_h[..., cutoff:] = 0
#
#        # Compute derivatives
#        results = []
#        for order in orders:
#            derivative_u_h = ((1j * k_x) ** order) * u_h
#            results.append(derivative_u_h)
#
#        derivatives_ft = torch.stack(results, dim=0)
#        derivatives_real = torch.fft.irfft(derivatives_ft, dim=-1, n=nx)
#
#        # Crop result if Fourier continuation was used
#        if self.use_fc:
#            start_x = self.fc_n_additional_pts // 2
#            end_x = start_x + u.shape[-1]
#            derivatives_real = derivatives_real[..., start_x:end_x]
#
#        return [derivatives_real[i] for i in range(len(orders))]
#
#
#    def derivative(self, u, order=1):
#        """
#        Compute the 1D Fourier derivative of a given tensor.
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input tensor. Expected shape: (..., length)
#        order : int, optional
#            Order of the derivative, by default 1
#
#        Returns
#        -------
#        torch.Tensor
#            The 1D derivative of the input tensor.
#        """
#        derivatives = self.compute_multiple_derivatives(u, [order])
#        return derivatives[0]
#
#
#
#
#
## class FourierDiff2D:
#    """
#    A class for computing 2D Fourier derivatives with Fourier continuation support.
#
#    Provides methods for computing partial and mixed 2D Fourier derivatives,
#    with optional Fourier continuation for handling non-periodic functions.
#    """
#
#    def __init__(self,
#                 L=(2*torch.pi, 2*torch.pi),
#                 use_fc=False, FC_d=4,
#                 FC_n_additional_pts=50,
#                 low_pass_filter_ratio=None):
#        """
#        Parameters
#        ----------
#        L : tuple or list, optional
#            Length of the domain along (y, x) directions, by default (2*pi, 2*pi)
#        use_fc : str, optional
#            Whether to use Fourier continuation. Use for non-periodic functions.
#            Options: None, 'Legendre', 'Gram', by default False
#        FC_d : int, optional
#            'Degree' of the Fourier continuation, by default 4
#        FC_n_additional_pts : int, optional
#            Number of points to add using the Fourier continuation layer, by default 50
#        low_pass_filter_ratio : float, optional
#            If not None, apply a low-pass filter to the Fourier coefficients, by default None
#        """
#        self.L = L
#        self.use_fc = use_fc
#        self.FC_d = FC_d
#        self.fc_n_additional_pts = FC_n_additional_pts
#        self.low_pass_filter_ratio = low_pass_filter_ratio
#
#
#    def compute_multiple_derivatives(self, u, derivatives):
#        """
#        Compute multiple derivatives in a single FFT/IFFT call for better performance.
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input tensor. Expected shape: (..., height, width)
#        derivatives : list of tuples
#            List of (order_x, order_y) tuples specifying which derivatives to compute.
#            Example: [(1, 0), (0, 1), (2, 0), (0, 2)] for dx, dy, dxx, dyy
#
#        Returns
#        -------
#        list of torch.Tensor
#            List of computed derivatives in the same order as the input derivatives list
#        """
#        if u is None:
#            raise ValueError("Input tensor u is None")
#
#        L_x, L_y = self.L[0], self.L[1]
#        nx, ny = u.shape[-2], u.shape[-1]
#        u_clone = u.clone()
#
#        # Apply Fourier continuation if specified
#        if self.use_fc == 'Legendre':
#            FC = FCLegendre(d=self.fc_degree, n_additional_pts=self.fc_n_additional_pts).to(u_clone.device)
#            u_clone = FC.extend2d(u_clone)
#            L_x *= (nx + self.fc_n_additional_pts) / nx
#            L_y *= (ny + self.fc_n_additional_pts) / ny
#        elif self.use_fc == 'Gram':
#            FC = FCGram(d=self.fc_degree, n_additional_pts=self.fc_n_additional_pts).to(u_clone.device)
#            u_clone = FC.extend2d(u_clone)
#            L_x *= (nx + self.fc_n_additional_pts) / nx
#            L_y *= (ny + self.fc_n_additional_pts) / ny
#
#        # Update grid parameters after extension
#        nx, ny = u_clone.shape[-2], u_clone.shape[-1]
#        dx, dy = L_x / nx, L_y / ny
#
#        # FFT with transposed axes (shape -> (ny, nx))
#        u_h = torch.fft.fft2(u_clone.transpose(-2, -1), dim=(-2, -1))
#
#        # Frequency arrays
#        k_x = torch.fft.fftfreq(nx, d=dx, device=u_h.device) * (2 * torch.pi)
#        k_y = torch.fft.fftfreq(ny, d=dy, device=u_h.device) * (2 * torch.pi)
#
#        # Create frequency meshgrid
#        KY, KX = torch.meshgrid(k_y, k_x, indexing="ij")
#
#        # Apply low-pass filter if specified
#        if self.low_pass_filter_ratio is not None:
#            cutoff_x = int(nx * self.low_pass_filter_ratio)
#            cutoff_y = int(ny * self.low_pass_filter_ratio)
#            u_h[..., cutoff_y:, :] = 0
#            u_h[..., :, cutoff_x:] = 0
#
#        # Compute derivatives
#        results = []
#        for order_x, order_y in derivatives:
#            # Expand meshgrid tensors for proper broadcasting
#            KX_expanded = KX.expand(u_h.shape)
#            KY_expanded = KY.expand(u_h.shape)
#
#            derivative_u_h = ((1j * KX_expanded) ** order_x) * ((1j * KY_expanded) ** order_y) * u_h
#            results.append(derivative_u_h)
#
#        derivatives_ft = torch.stack(results, dim=0)
#        derivatives_real = torch.fft.ifft2(derivatives_ft, dim=(-2, -1)).real
#
#        # Transpose back to original shape (nx, ny)
#        derivatives_real = derivatives_real.transpose(-2, -1)
#
#        # Crop result if Fourier continuation was used
#        if self.use_fc:
#            start_x = self.fc_n_additional_pts // 2
#            end_x = start_x + u.shape[-2]
#            start_y = self.fc_n_additional_pts // 2
#            end_y = start_y + u.shape[-1]
#            derivatives_real = derivatives_real[..., start_x:end_x, start_y:end_y]
#
#        return [derivatives_real[i] for i in range(len(derivatives))]
#
#
#    def derivative(self, u, order_x=1, order_y=1):
#        """
#        Compute the 2D Fourier derivative of a given tensor.
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input tensor. Expected shape: (..., height, width)
#        order_x : int, optional
#            Order of the derivative along the x-direction (last dimension), by default 1
#        order_y : int, optional
#            Order of the derivative along the y-direction (second-to-last dimension), by default 1
#
#        Returns
#        -------
#        torch.Tensor
#            The 2D derivative of the input tensor.
#        """
#        derivatives = self.compute_multiple_derivatives(u, [(order_x, order_y)])
#        return derivatives[0]
#
#
#    def partial(self, u, direction='x', order=1):
#        """
#        Compute partial 2D Fourier derivative along a specific direction.
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input tensor. Expected shape: (..., height, width)
#        direction : str, optional
#            Direction of differentiation: 'x' (last dimension) or 'y' (second-to-last dimension), by default 'x'
#        order : int, optional
#            Order of the derivative, by default 1
#
#        Returns
#        -------
#        torch.Tensor
#            The partial derivative of the input tensor along the specified direction.
#        """
#        if direction == 'x':
#            return self.derivative(u, order_x=order, order_y=0)
#        elif direction == 'y':
#            return self.derivative(u, order_x=0, order_y=order)
#        else:
#            raise ValueError("Direction must be 'x' or 'y'")
#
#
#    def dx(self, u, order=1):
#        """
#        Compute partial derivative with respect to x.
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input tensor
#        order : int, optional
#            Order of the derivative, by default 1
#
#        Returns
#        -------
#        torch.Tensor
#            The partial derivative with respect to x
#        """
#        return self.partial(u, direction='x', order=order)
#
#
#    def dy(self, u, order=1):
#        """
#        Compute partial derivative with respect to y.
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input tensor
#        order : int, optional
#            Order of the derivative, by default 1
#
#        Returns
#        -------
#        torch.Tensor
#            The partial derivative with respect to y
#        """
#        return self.partial(u, direction='y', order=order)
#
#
#    def laplacian(self, u):
#        """
#        Compute the 2D Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y².
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input tensor
#
#        Returns
#        -------
#        torch.Tensor
#            The Laplacian of the input tensor
#        """
#        dxx, dyy = self.compute_multiple_derivatives(u, [(2, 0), (0, 2)])
#        return dxx + dyy
#
#
#    def divergence(self, u):
#        """
#        Compute the 2D divergence ∇·u = ∂u₁/∂x + ∂u₂/∂y.
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input vector field with shape (..., 2, height, width)
#
#        Returns
#        -------
#        torch.Tensor
#            The divergence of the vector field
#        """
#        if u.shape[-3] != 2:
#            raise ValueError("Input must be a 2D vector field with shape (..., 2, height, width)")
#
#        # Extract components and ensure they have the right shape
#        u1 = u[..., 0, :, :].contiguous()  # First component
#        u2 = u[..., 1, :, :].contiguous()  # Second component
#
#        # Ensure the components are not None and have the right shape
#        if u1 is None or u2 is None:
#            raise ValueError("Vector field components cannot be None")
#
#        return self.dx(u1) + self.dy(u2)
#
#
#    def curl(self, u):
#        """
#        Compute the 2D curl ∇×u = ∂u₂/∂x - ∂u₁/∂y.
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input vector field with shape (..., 2, height, width)
#
#        Returns
#        -------
#        torch.Tensor
#            The curl of the vector field (scalar field in 2D)
#        """
#        if u.shape[-3] != 2:
#            raise ValueError("Input must be a 2D vector field with shape (..., 2, height, width)")
#
#        u1 = u[..., 0, :, :]
#        u2 = u[..., 1, :, :]
#
#        return self.dx(u2) - self.dy(u1)
#
#
#    def gradient(self, u):
#        """
#        Compute the 2D gradient ∇f = [∂f/∂x, ∂f/∂y].
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input scalar field
#
#        Returns
#        -------
#        torch.Tensor
#            The gradient of the scalar field with shape (..., 2, height, width)
#        """
#        grad_x = self.dx(u)
#        grad_y = self.dy(u)
#
#        return torch.stack([grad_x, grad_y], dim=-3)
#
#
## class FourierDiff3D:#
#    """
#    A class for computing 3D Fourier derivatives with Fourier continuation support.
#
#    Provides methods for computing partial and mixed 3D Fourier derivatives,
#    with optional Fourier continuation for handling non-periodic functions.
#    """
#
#    def __init__(self,
#                 L=(2*torch.pi, 2*torch.pi, 2*torch.pi),
#                 use_fc=False, FC_d=4,
#                 FC_n_additional_pts=50,
#                 low_pass_filter_ratio=None):
#        """
#        Parameters
#        ----------
#        L : tuple or list, optional
#            Length of the domain along (z, y, x) directions, by default (2*pi, 2*pi, 2*pi)
#        use_fc : str, optional
#            Whether to use Fourier continuation. Use for non-periodic functions.
#            Options: None, 'Legendre', 'Gram', by default False
#        FC_d : int, optional
#            'Degree' of the Fourier continuation, by default 4
#        FC_n_additional_pts : int, optional
#            Number of points to add using the Fourier continuation layer, by default 50
#        low_pass_filter_ratio : float, optional
#            If not None, apply a low-pass filter to the Fourier coefficients, by default None
#        """
#        self.L = L
#        self.use_fc = use_fc
#        self.FC_d = FC_d
#        self.fc_n_additional_pts = FC_n_additional_pts
#        self.low_pass_filter_ratio = low_pass_filter_ratio
#
#
#    def compute_multiple_derivatives(self, u, derivatives):
#        """
#        Compute multiple derivatives in a single FFT/IFFT call for better performance.
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input tensor. Expected shape: (..., width, height, depth)
#        derivatives : list of tuples
#            List of (order_x, order_y, order_z) tuples specifying which derivatives to compute.
#            Example: [(1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (0, 2, 0), (0, 0, 2)]
#            for dx, dy, dz, dxx, dyy, dzz
#
#        Returns
#        -------
#        list of torch.Tensor
#            List of computed derivatives in the same order as the input derivatives list
#        """
#        if u is None:
#            raise ValueError("Input tensor u is None")
#
#        L_x, L_y, L_z = self.L[0], self.L[1], self.L[2]
#        nx, ny, nz = u.shape[-3], u.shape[-2], u.shape[-1]
#        u_clone = u.clone()
#
#        # Apply Fourier continuation if specified
#        if self.use_fc == 'Legendre':
#            FC = FCLegendre(d=self.fc_degree, n_additional_pts=self.fc_n_additional_pts).to(u_clone.device)
#            u_clone = FC.extend3d(u_clone)
#            L_x *= (nx + self.fc_n_additional_pts) / nx
#            L_y *= (ny + self.fc_n_additional_pts) / ny
#            L_z *= (nz + self.fc_n_additional_pts) / nz
#        elif self.use_fc == 'Gram':
#            FC = FCGram(d=self.fc_degree, n_additional_pts=self.fc_n_additional_pts).to(u_clone.device)
#            u_clone = FC.extend3d(u_clone)
#            L_x *= (nx + self.fc_n_additional_pts) / nx
#            L_y *= (ny + self.fc_n_additional_pts) / ny
#            L_z *= (nz + self.fc_n_additional_pts) / nz
#
#        # Update grid parameters after extension
#        nx, ny, nz = u_clone.shape[-3], u_clone.shape[-2], u_clone.shape[-1]
#        dx, dy, dz = L_x / nx, L_y / ny, L_z / nz
#
#        # FFT with permuted axes (shape -> (nz, ny, nx))
#        u_clone_permuted = u_clone.permute(*range(u_clone.ndim-3), -1, -2, -3)
#        u_h = torch.fft.fftn(u_clone_permuted, dim=(-3, -2, -1))
#
#        # Frequency arrays
#        k_x = torch.fft.fftfreq(nx, d=dx, device=u_h.device) * (2 * torch.pi)
#        k_y = torch.fft.fftfreq(ny, d=dy, device=u_h.device) * (2 * torch.pi)
#        k_z = torch.fft.fftfreq(nz, d=dz, device=u_h.device) * (2 * torch.pi)
#
#        # Create frequency meshgrid
#        KZ, KY, KX = torch.meshgrid(k_z, k_y, k_x, indexing="ij")
#
#        # Apply low-pass filter if specified
#        if self.low_pass_filter_ratio is not None:
#            cutoff_x = int(nx * self.low_pass_filter_ratio)
#            cutoff_y = int(ny * self.low_pass_filter_ratio)
#            cutoff_z = int(nz * self.low_pass_filter_ratio)
#            u_h[..., cutoff_y:, :, :] = 0
#            u_h[..., :, cutoff_x:, :] = 0
#            u_h[..., :, :, cutoff_z:] = 0
#
#
#        # Compute derivatives
#        results = []
#        for order_x, order_y, order_z in derivatives:
#            # Expand meshgrid tensors for proper broadcasting
#            KX_expanded = KX.expand(u_h.shape)
#            KY_expanded = KY.expand(u_h.shape)
#            KZ_expanded = KZ.expand(u_h.shape)
#
#            derivative_u_h = ((1j * KX_expanded) ** order_x) * ((1j * KY_expanded) ** order_y) * ((1j * KZ_expanded) ** order_z) * u_h
#            results.append(derivative_u_h)
#
#        derivatives_ft = torch.stack(results, dim=0)
#        derivatives_real = torch.fft.ifftn(derivatives_ft, dim=(-3, -2, -1)).real
#
#        # Permute back to original shape (..., nx, ny, nz)
#        derivatives_real = derivatives_real.permute(*range(derivatives_real.ndim-3), -1, -2, -3)
#
#        # Crop result if Fourier continuation was used
#        if self.use_fc:
#            start_x = self.fc_n_additional_pts // 2
#            end_x = start_x + u.shape[-3]
#            start_y = self.fc_n_additional_pts // 2
#            end_y = start_y + u.shape[-2]
#            start_z = self.fc_n_additional_pts // 2
#            end_z = start_z + u.shape[-1]
#            derivatives_real = derivatives_real[..., start_x:end_x, start_y:end_y, start_z:end_z]
#
#        return [derivatives_real[i] for i in range(len(derivatives))]
#
#
#    def derivative(self, u, order_x=1, order_y=1, order_z=1):
#        """
#        Compute the 3D Fourier derivative of a given tensor.
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input tensor. Expected shape: (..., width, height, depth)
#        order_x : int, optional
#            Order of the derivative along the x-direction (last dimension), by default 1
#        order_y : int, optional
#            Order of the derivative along the y-direction (second-to-last dimension), by default 1
#        order_z : int, optional
#            Order of the derivative along the z-direction (third-to-last dimension), by default 1
#
#        Returns
#        -------
#        torch.Tensor
#            The 3D derivative of the input tensor.
#        """
#        derivatives = self.compute_multiple_derivatives(u, [(order_x, order_y, order_z)])
#        return derivatives[0]
#
#
#    def partial(self, u, direction='x', order=1):
#        """
#        Compute partial 3D Fourier derivative along a specific direction.
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input tensor. Expected shape: (..., width, height, depth)
#        direction : str, optional
#            Direction of differentiation: 'x' (last dimension), 'y' (second-to-last dimension),
#            or 'z' (third-to-last dimension), by default 'x'
#        order : int, optional
#            Order of the derivative, by default 1
#
#        Returns
#        -------
#        torch.Tensor
#            The partial derivative of the input tensor along the specified direction.
#        """
#        if direction == 'x':
#            derivatives = self.compute_multiple_derivatives(u, [(order, 0, 0)])
#            return derivatives[0]
#        elif direction == 'y':
#            derivatives = self.compute_multiple_derivatives(u, [(0, order, 0)])
#            return derivatives[0]
#        elif direction == 'z':
#            derivatives = self.compute_multiple_derivatives(u, [(0, 0, order)])
#            return derivatives[0]
#        else:
#            raise ValueError("Direction must be 'x', 'y', or 'z'")
#
#
#    def dx(self, u, order=1):
#        """
#        Compute partial derivative with respect to x.
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input tensor
#        order : int, optional
#            Order of the derivative, by default 1
#
#        Returns
#        -------
#        torch.Tensor
#            The partial derivative with respect to x
#        """
#        return self.partial(u, direction='x', order=order)
#
#
#    def dy(self, u, order=1):
#        """
#        Compute partial derivative with respect to y.
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input tensor
#        order : int, optional
#            Order of the derivative, by default 1
#
#        Returns
#        -------
#        torch.Tensor
#            The partial derivative with respect to y
#        """
#        return self.partial(u, direction='y', order=order)
#
#
#    def dz(self, u, order=1):
#        """
#        Compute partial derivative with respect to z.
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input tensor
#        order : int, optional
#            Order of the derivative, by default 1
#
#        Returns
#        -------
#        torch.Tensor
#            The partial derivative with respect to z
#        """
#        return self.partial(u, direction='z', order=order)
#
#
#    def laplacian(self, u):
#        """
#        Compute the 3D Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z².
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input tensor
#
#        Returns
#        -------
#        torch.Tensor
#            The Laplacian of the input tensor
#        """
#        # Use efficient multiple derivatives computation
#        dxx, dyy, dzz = self.compute_multiple_derivatives(u, [(2, 0, 0), (0, 2, 0), (0, 0, 2)])
#        return dxx + dyy + dzz
#
#
#    def divergence(self, u):
#        """
#        Compute the 3D divergence ∇·u = ∂u₁/∂x + ∂u₂/∂y + ∂u₃/∂z.
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input vector field with shape (..., 3, width, height, depth)
#
#        Returns
#        -------
#        torch.Tensor
#            The divergence of the vector field
#        """
#        if u.shape[-4] != 3:
#            raise ValueError("Input must be a 3D vector field with shape (..., 3, width, height, depth)")
#
#        # Extract components and ensure they have the right shape
#        u1 = u[..., 0, :, :, :].contiguous()  # First component
#        u2 = u[..., 1, :, :, :].contiguous()  # Second component
#        u3 = u[..., 2, :, :, :].contiguous()  # Third component
#
#        # Ensure the components are not None and have the right shape
#        if u1 is None or u2 is None or u3 is None:
#            raise ValueError("Vector field components cannot be None")
#
#        return self.dx(u1) + self.dy(u2) + self.dz(u3)
#
#
#    def curl(self, u):
#        """
#        Compute the 3D curl ∇×u = [∂u₃/∂y - ∂u₂/∂z, ∂u₁/∂z - ∂u₃/∂x, ∂u₂/∂x - ∂u₁/∂y].
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input vector field with shape (..., 3, width, height, depth)
#
#        Returns
#        -------
#        torch.Tensor
#            The curl of the vector field with shape (..., 3, width, height, depth)
#        """
#        if u.shape[-4] != 3:
#            raise ValueError("Input must be a 3D vector field with shape (..., 3, width, height, depth)")
#
#        u1 = u[..., 0, :, :, :]  # First component
#        u2 = u[..., 1, :, :, :]  # Second component
#        u3 = u[..., 2, :, :, :]  # Third component
#
#        curl_x = self.dy(u3) - self.dz(u2)
#        curl_y = self.dz(u1) - self.dx(u3)
#        curl_z = self.dx(u2) - self.dy(u1)
#
#        return torch.stack([curl_x, curl_y, curl_z], dim=-4)
#
#
#    def gradient(self, u):
#        """
#        Compute the 3D gradient ∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z].
#
#        Parameters
#        ----------
#        u : torch.Tensor
#            Input scalar field
#
#        Returns
#        -------
#        torch.Tensor
#            The gradient of the scalar field with shape (..., 3, width, height, depth)
#        """
#        grad_x = self.dx(u)
#        grad_y = self.dy(u)
#        grad_z = self.dz(u)
#        
#        return torch.stack([grad_x, grad_y, grad_z], dim=-4)
