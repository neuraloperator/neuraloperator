from ..layers.fourier_continuation import FCLegendre, FCGram
import torch
import warnings


def fourier_derivative_1d(u, order=1, L=2*torch.pi, use_FC=False, FC_d=4, FC_n_additional_pts=50, low_pass_filter_ratio=None):
    """
    Compute the 1D Fourier derivative of a given tensor.
    Use with care, as Fourier continuation and Fourier derivatives are not always stable.
    
    FC derivatives are more stable when the original signal is nearly periodic 
        (i.e., values and slopes at boundaries match or are close).
        and when the signal is smooth and has no sharp jumps/discontinuities at the boundaries.
    
    Use Fourier continuation to extend the signal if it is non-periodic. 
    
    Unstable behavior can occur if the signal has strong discontinuities or non-matching derivatives at boundaries,
        and if the continuation introduces artificial oscillations (Gibbs phenomenon).

    Signs of instability include boundary artifacts, high-frequency ringing, or growing errors with higher resolution.
    
    
    Parameters
    ----------
    u : torch.Tensor
        Input tensor. The derivative will be computed along the last dimension.
    order : int, optional
        Order of the derivative, by default 1
    L : float, optional
        Length of the domain considered, by default 2*pi
    use_FC : str, optional
        Whether to use Fourier continuation. Use for non-periodic functions.
        Options: None, 'Legendre', 'Gram', by default False
    FC_d : int, optional
        'Degree' of the Fourier continuation, by default 4
    FC_n_additional_pts : int, optional
        Number of points to add using the Fourier continuation layer, by default 50
        For FC-Gram continuation, it is usually not necessary to change this parameter. 
        This has a bigger effect on FC-Legendre continuation.
    low_pass_filter_ratio : float, optional
        If not None, apply a low-pass filter to the Fourier coefficients. 
        Can help reduce artificial oscillations. 1.0 means no filtering, 
        0.5 means keep half of the coefficients, etc., by default None

    Returns
    -------
    torch.Tensor
        The derivative of the input tensor.
        
    Notes
    -----
    When using Fourier continuation, the function automatically adjusts the 
    domain length L to account for the extended signal. The result is cropped 
    back to the original interval size.
    
    Warnings
    --------
    Consider using Fourier continuation if the input is not periodic (use_FC=True).
    Fourier continuation and Fourier derivatives can be numerically unstable
    for certain functions and parameter combinations.
    """
    
    # Extend signal using Fourier continuation if specified
    if use_FC == 'Legendre':
        L = L * (u.shape[-1] + FC_n_additional_pts) / u.shape[-1]  # Define extended length
        FC = FCLegendre(d=FC_d, n_additional_pts=FC_n_additional_pts).to(u.device)
        u = FC(u, dim=1)
    elif use_FC == 'Gram':
        L = L * (u.shape[-1] + FC_n_additional_pts) / u.shape[-1]  # Define extended length
        FC = FCGram(d=FC_d, n_additional_pts=FC_n_additional_pts).to(u.device)
        u = FC(u, dim=1)
    else:
        warnings.warn("Consider using Fourier continuation if the input is not periodic (use_FC=True).", category=UserWarning)

    nx = u.size(-1)
    dx = L / nx
    u_h = torch.fft.rfft(u, dim=-1)
    k_x = torch.fft.rfftfreq(nx, d=dx, device=u_h.device) * (2 * torch.pi)
    
    if low_pass_filter_ratio is not None:
        # Apply low-pass filter to Fourier coefficients
        cutoff = int(u_h.shape[-1] * low_pass_filter_ratio)
        u_h[..., cutoff:] = 0
    
    # Compute Fourier derivative
    derivative_u_h = (1j * k_x)**order * u_h
    
    # Transform back to physical space
    derivative_u = torch.fft.irfft(derivative_u_h, dim=-1, n=nx) 

    # Crop result if Fourier continuation was used
    if use_FC:
        derivative_u = derivative_u[..., FC_n_additional_pts//2: -FC_n_additional_pts//2]

    return derivative_u


class FourierDiff2D:
    """
    A class for computing 2D Fourier derivatives with Fourier continuation support.
    
    Provides methods for computing partial and mixed 2D Fourier derivatives,
    with optional Fourier continuation for handling non-periodic functions.
    """
    
    def __init__(self, 
                 L=(2*torch.pi, 2*torch.pi), 
                 use_FC=False, FC_d=4, 
                 FC_n_additional_pts=50, 
                 low_pass_filter_ratio=None):
        """
        Parameters
        ----------
        L : tuple or list, optional
            Length of the domain along (y, x) directions, by default (2*pi, 2*pi)
        use_FC : str, optional
            Whether to use Fourier continuation. Use for non-periodic functions.
            Options: None, 'Legendre', 'Gram', by default False
        FC_d : int, optional
            'Degree' of the Fourier continuation, by default 4
        FC_n_additional_pts : int, optional
            Number of points to add using the Fourier continuation layer, by default 50
        low_pass_filter_ratio : float, optional
            If not None, apply a low-pass filter to the Fourier coefficients, by default None
        """
        self.L = L
        self.use_FC = use_FC
        self.FC_d = FC_d
        self.FC_n_additional_pts = FC_n_additional_pts
        self.low_pass_filter_ratio = low_pass_filter_ratio

    
    def compute_multiple_derivatives(self, u, derivatives):
        """
        Compute multiple derivatives in a single FFT/IFFT call for better performance.
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor. Expected shape: (..., height, width)
        derivatives : list of tuples
            List of (order_x, order_y) tuples specifying which derivatives to compute.
            Example: [(1, 0), (0, 1), (2, 0), (0, 2)] for dx, dy, dxx, dyy
            
        Returns
        -------
        list of torch.Tensor
            List of computed derivatives in the same order as the input derivatives list
        """
        if u is None:
            raise ValueError("Input tensor u is None")

        L_x, L_y = self.L[0], self.L[1]
        nx, ny = u.shape[-2], u.shape[-1]
        u_work = u.clone()

        # Apply Fourier continuation if specified
        if self.use_FC == 'Legendre':
            FC = FCLegendre(d=self.FC_d, n_additional_pts=self.FC_n_additional_pts).to(u_work.device)
            u_work = FC.extend2d(u_work)
            L_x *= (nx + self.FC_n_additional_pts) / nx
            L_y *= (ny + self.FC_n_additional_pts) / ny
        elif self.use_FC == 'Gram':
            FC = FCGram(d=self.FC_d, n_additional_pts=self.FC_n_additional_pts).to(u_work.device)
            u_work = FC.extend2d(u_work)
            L_x *= (nx + self.FC_n_additional_pts) / nx
            L_y *= (ny + self.FC_n_additional_pts) / ny

        # Update grid parameters after extension
        nx, ny = u_work.shape[-2], u_work.shape[-1]
        dx, dy = L_x / nx, L_y / ny

        # FFT with transposed axes (shape -> (ny, nx))
        u_h = torch.fft.fft2(u_work.transpose(-2, -1), dim=(-2, -1))

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
        if self.use_FC:
            start_x = self.FC_n_additional_pts // 2
            end_x = start_x + u.shape[-2]
            start_y = self.FC_n_additional_pts // 2
            end_y = start_y + u.shape[-1]
            derivatives_real = derivatives_real[..., start_x:end_x, start_y:end_y]

        return [derivatives_real[i] for i in range(len(derivatives))]
    
    
    def derivative(self, u, order_x=1, order_y=1):
        """
        Compute the 2D Fourier derivative of a given tensor.
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor. Expected shape: (..., height, width)
        order_x : int, optional
            Order of the derivative along the x-direction (last dimension), by default 1
        order_y : int, optional
            Order of the derivative along the y-direction (second-to-last dimension), by default 1
            
        Returns
        -------
        torch.Tensor
            The 2D derivative of the input tensor.
        """
        derivatives = self.compute_multiple_derivatives(u, [(order_x, order_y)])
        return derivatives[0]
    
    
    def partial(self, u, direction='x', order=1):
        """
        Compute partial 2D Fourier derivative along a specific direction.
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor. Expected shape: (..., height, width)
        direction : str, optional
            Direction of differentiation: 'x' (last dimension) or 'y' (second-to-last dimension), by default 'x'
        order : int, optional
            Order of the derivative, by default 1
            
        Returns
        -------
        torch.Tensor
            The partial derivative of the input tensor along the specified direction.
        """
        if direction == 'x':
            return self.derivative(u, order_x=order, order_y=0)
        elif direction == 'y':
            return self.derivative(u, order_x=0, order_y=order)
        else:
            raise ValueError("Direction must be 'x' or 'y'")
    
    
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
            The partial derivative with respect to x
        """
        return self.partial(u, direction='x', order=order)
    
    
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
            The partial derivative with respect to y
        """
        return self.partial(u, direction='y', order=order)
    
    
    def laplacian(self, u):
        """
        Compute the 2D Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y².
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor
            
        Returns
        -------
        torch.Tensor
            The Laplacian of the input tensor
        """
        dxx, dyy = self.compute_multiple_derivatives(u, [(2, 0), (0, 2)])
        return dxx + dyy
    
    
    def divergence(self, u):
        """
        Compute the 2D divergence ∇·u = ∂u₁/∂x + ∂u₂/∂y.
        
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
            raise ValueError("Input must be a 2D vector field with shape (..., 2, height, width)")
        
        # Extract components and ensure they have the right shape
        u1 = u[..., 0, :, :].contiguous()  # First component
        u2 = u[..., 1, :, :].contiguous()  # Second component
        
        # Ensure the components are not None and have the right shape
        if u1 is None or u2 is None:
            raise ValueError("Vector field components cannot be None")
        
        return self.dx(u1) + self.dy(u2)
    
    
    def curl(self, u):
        """
        Compute the 2D curl ∇×u = ∂u₂/∂x - ∂u₁/∂y.
        
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
            raise ValueError("Input must be a 2D vector field with shape (..., 2, height, width)")
        
        u1 = u[..., 0, :, :] 
        u2 = u[..., 1, :, :]  
        
        return self.dx(u2) - self.dy(u1)
    
    
    def gradient(self, u):
        """
        Compute the 2D gradient ∇f = [∂f/∂x, ∂f/∂y].
        
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


class FourierDiff3D:
    """
    A class for computing 3D Fourier derivatives with Fourier continuation support.
    
    Provides methods for computing partial and mixed 3D Fourier derivatives,
    with optional Fourier continuation for handling non-periodic functions.
    """
    
    def __init__(self, 
                 L=(2*torch.pi, 2*torch.pi, 2*torch.pi), 
                 use_FC=False, FC_d=4, 
                 FC_n_additional_pts=50, 
                 low_pass_filter_ratio=None):
        """
        Parameters
        ----------
        L : tuple or list, optional
            Length of the domain along (z, y, x) directions, by default (2*pi, 2*pi, 2*pi)
        use_FC : str, optional
            Whether to use Fourier continuation. Use for non-periodic functions.
            Options: None, 'Legendre', 'Gram', by default False
        FC_d : int, optional
            'Degree' of the Fourier continuation, by default 4
        FC_n_additional_pts : int, optional
            Number of points to add using the Fourier continuation layer, by default 50
        low_pass_filter_ratio : float, optional
            If not None, apply a low-pass filter to the Fourier coefficients, by default None
        """
        self.L = L
        self.use_FC = use_FC
        self.FC_d = FC_d
        self.FC_n_additional_pts = FC_n_additional_pts
        self.low_pass_filter_ratio = low_pass_filter_ratio
        
        
    def compute_multiple_derivatives(self, u, derivatives):
        """
        Compute multiple derivatives in a single FFT/IFFT call for better performance.
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor. Expected shape: (..., width, height, depth)
        derivatives : list of tuples
            List of (order_x, order_y, order_z) tuples specifying which derivatives to compute.
            Example: [(1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (0, 2, 0), (0, 0, 2)] 
            for dx, dy, dz, dxx, dyy, dzz
            
        Returns
        -------
        list of torch.Tensor
            List of computed derivatives in the same order as the input derivatives list
        """
        if u is None:
            raise ValueError("Input tensor u is None")

        L_x, L_y, L_z = self.L[0], self.L[1], self.L[2]
        nx, ny, nz = u.shape[-3], u.shape[-2], u.shape[-1]
        u_work = u.clone()

        # Apply Fourier continuation if specified
        if self.use_FC == 'Legendre':
            FC = FCLegendre(d=self.FC_d, n_additional_pts=self.FC_n_additional_pts).to(u_work.device)
            u_work = FC.extend3d(u_work)
            L_x *= (nx + self.FC_n_additional_pts) / nx
            L_y *= (ny + self.FC_n_additional_pts) / ny
            L_z *= (nz + self.FC_n_additional_pts) / nz
        elif self.use_FC == 'Gram':
            FC = FCGram(d=self.FC_d, n_additional_pts=self.FC_n_additional_pts).to(u_work.device)
            u_work = FC.extend3d(u_work)
            L_x *= (nx + self.FC_n_additional_pts) / nx
            L_y *= (ny + self.FC_n_additional_pts) / ny
            L_z *= (nz + self.FC_n_additional_pts) / nz

        # Update grid parameters after extension
        nx, ny, nz = u_work.shape[-3], u_work.shape[-2], u_work.shape[-1]
        dx, dy, dz = L_x / nx, L_y / ny, L_z / nz

        # FFT with permuted axes (shape -> (nz, ny, nx))
        u_work_permuted = u_work.permute(*range(u_work.ndim-3), -1, -2, -3)
        u_h = torch.fft.fftn(u_work_permuted, dim=(-3, -2, -1))

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
        if self.use_FC:
            start_x = self.FC_n_additional_pts // 2
            end_x = start_x + u.shape[-3]
            start_y = self.FC_n_additional_pts // 2
            end_y = start_y + u.shape[-2]
            start_z = self.FC_n_additional_pts // 2
            end_z = start_z + u.shape[-1]
            derivatives_real = derivatives_real[..., start_x:end_x, start_y:end_y, start_z:end_z]

        return [derivatives_real[i] for i in range(len(derivatives))]
       
       
    def derivative(self, u, order_x=1, order_y=1, order_z=1):
        """
        Compute the 3D Fourier derivative of a given tensor.
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor. Expected shape: (..., width, height, depth)
        order_x : int, optional
            Order of the derivative along the x-direction (last dimension), by default 1
        order_y : int, optional
            Order of the derivative along the y-direction (second-to-last dimension), by default 1
        order_z : int, optional
            Order of the derivative along the z-direction (third-to-last dimension), by default 1
            
        Returns
        -------
        torch.Tensor
            The 3D derivative of the input tensor.
        """
        derivatives = self.compute_multiple_derivatives(u, [(order_x, order_y, order_z)])
        return derivatives[0]
    
    
    def partial(self, u, direction='x', order=1):
        """
        Compute partial 3D Fourier derivative along a specific direction.
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor. Expected shape: (..., width, height, depth)
        direction : str, optional
            Direction of differentiation: 'x' (last dimension), 'y' (second-to-last dimension), 
            or 'z' (third-to-last dimension), by default 'x'
        order : int, optional
            Order of the derivative, by default 1
            
        Returns
        -------
        torch.Tensor
            The partial derivative of the input tensor along the specified direction.
        """
        if direction == 'x':
            derivatives = self.compute_multiple_derivatives(u, [(order, 0, 0)])
            return derivatives[0]
        elif direction == 'y':
            derivatives = self.compute_multiple_derivatives(u, [(0, order, 0)])
            return derivatives[0]
        elif direction == 'z':
            derivatives = self.compute_multiple_derivatives(u, [(0, 0, order)])
            return derivatives[0]
        else:
            raise ValueError("Direction must be 'x', 'y', or 'z'")
    
    
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
            The partial derivative with respect to x
        """
        return self.partial(u, direction='x', order=order)
    
    
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
            The partial derivative with respect to y
        """
        return self.partial(u, direction='y', order=order)
    
    
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
            The partial derivative with respect to z
        """
        return self.partial(u, direction='z', order=order)
    
    
    def laplacian(self, u):
        """
        Compute the 3D Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z².
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor
            
        Returns
        -------
        torch.Tensor
            The Laplacian of the input tensor
        """
        # Use efficient multiple derivatives computation
        dxx, dyy, dzz = self.compute_multiple_derivatives(u, [(2, 0, 0), (0, 2, 0), (0, 0, 2)])
        return dxx + dyy + dzz
    
    
    def divergence(self, u):
        """
        Compute the 3D divergence ∇·u = ∂u₁/∂x + ∂u₂/∂y + ∂u₃/∂z.
        
        Parameters
        ----------
        u : torch.Tensor
            Input vector field with shape (..., 3, width, height, depth)
            
        Returns
        -------
        torch.Tensor
            The divergence of the vector field
        """
        if u.shape[-4] != 3:
            raise ValueError("Input must be a 3D vector field with shape (..., 3, width, height, depth)")
        
        # Extract components and ensure they have the right shape
        u1 = u[..., 0, :, :, :].contiguous()  # First component
        u2 = u[..., 1, :, :, :].contiguous()  # Second component
        u3 = u[..., 2, :, :, :].contiguous()  # Third component
        
        # Ensure the components are not None and have the right shape
        if u1 is None or u2 is None or u3 is None:
            raise ValueError("Vector field components cannot be None")
        
        return self.dx(u1) + self.dy(u2) + self.dz(u3)
    
    
    def curl(self, u):
        """
        Compute the 3D curl ∇×u = [∂u₃/∂y - ∂u₂/∂z, ∂u₁/∂z - ∂u₃/∂x, ∂u₂/∂x - ∂u₁/∂y].
        
        Parameters
        ----------
        u : torch.Tensor
            Input vector field with shape (..., 3, width, height, depth)
            
        Returns
        -------
        torch.Tensor
            The curl of the vector field with shape (..., 3, width, height, depth)
        """
        if u.shape[-4] != 3:
            raise ValueError("Input must be a 3D vector field with shape (..., 3, width, height, depth)")
        
        u1 = u[..., 0, :, :, :]  # First component
        u2 = u[..., 1, :, :, :]  # Second component
        u3 = u[..., 2, :, :, :]  # Third component
        
        curl_x = self.dy(u3) - self.dz(u2)
        curl_y = self.dz(u1) - self.dx(u3)
        curl_z = self.dx(u2) - self.dy(u1)
        
        return torch.stack([curl_x, curl_y, curl_z], dim=-4)
    
    
    def gradient(self, u):
        """
        Compute the 3D gradient ∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z].
        
        Parameters
        ----------
        u : torch.Tensor
            Input scalar field
            
        Returns
        -------
        torch.Tensor
            The gradient of the scalar field with shape (..., 3, width, height, depth)
        """
        grad_x = self.dx(u)
        grad_y = self.dy(u)
        grad_z = self.dz(u)
        
        return torch.stack([grad_x, grad_y, grad_z], dim=-4)