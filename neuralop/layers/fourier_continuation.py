import torch
import torch.nn as nn
import numpy as np
from numpy.polynomial.legendre import Legendre
import warnings
from pathlib import Path


class FourierContinuation(nn.Module):
    """
    Base class for Fourier Continuation implementations.
    
    This class provides the common interface and methods for Fourier Continuation
    using different approaches (Legendre polynomials, Gram matrices, etc.).
    """
    
    def __init__(self, d=5, n_additional_pts=50):
        """
        Initialize FourierContinuation with specified parameters.
        
        Parameters
        ----------
        d : int
            Number of matching points on the left and right boundaries
        n_additional_pts : int
            Number of additional points to add for continuation
        """
        super().__init__()
        
        self.d = d
        self.n_additional_pts = n_additional_pts


    def extend(self, x, dim=2):
        """
        Extend tensor along specified dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        dim : int or tuple of ints
            If int: extend along last dim axes
            If tuple: extend along the given axes (supports negative indexing)
        """
        # Convert input dimension(s) to list of axes to extend along:
        # If dim is an integer n, extend along the last n dimensions
        if isinstance(dim, int):
            axes = list(range(-dim, 0))
        # If dim is a tuple, extend along those specific dimensions
        else:
            # Convert positive indices to negative indices for consistency
            axes = [a if a < 0 else a - x.ndim for a in dim]

        # Extend along each axis
        for axis in axes:
            x = self.extend_along_axis(x, axis)
            
        return x

    def forward(self, x, dim=2):
        """
        Forward pass that calls the extend method.
        
        This allows the module to be used as a standard PyTorch module.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        dim : int or tuple of ints
            If int: extend along last dim axes
            If tuple: extend along the given axes (supports negative indexing)
            
        Returns
        -------
        torch.Tensor
            Extended tensor
        """
        return self.extend(x, dim)

    def extend_along_axis(self, x, axis):
        """
        Extend function values along a specific axis.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        axis : int
            Axis along which to extend (supports negative indexing)
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        # Convert negative axis to positive
        if axis < 0:
            axis = x.ndim + axis
        
        # If already extending along the last axis, use direct method
        if axis == x.ndim - 1:
            return self._extend_last_axis(x)
        
        # Otherwise, permute to move axis to last position
        axes = list(range(x.ndim))
        axes[axis], axes[-1] = axes[-1], axes[axis]  # Swap target axis with last axis
        
        # Permute tensor and extend along the (now last) axis
        x_swapped = x.permute(axes)
        x_extended = self._extend_last_axis(x_swapped)
        
        # Create inverse permutation to restore original axis order
        inverse_axes = list(range(len(axes)))
        inverse_axes = [axes.index(i) for i in range(len(axes))]
        
        # Permute back to original order
        return x_extended.permute(inverse_axes)
    
    

    def restrict(self, x, dim):
        """
        Remove Fourier continuation extension points to restore original domain size.
        
        Reverses the extension process by removing half of the additional points
        on each side that were added during Fourier continuation.
        
        Parameters
        ----------
        x : torch.Tensor
            Extended tensor from Fourier continuation
        dim : int or tuple of ints
            If int: restrict along last `dim` axes
            If tuple: restrict along the given axes (supports negative indexing)
        
        Returns
        -------
        torch.Tensor
            Tensor with original domain size, half of extension points removed from each side
        """
        # Convert input dimension(s) to list of axes to restrict along:
        # If dim is an integer n, restrict along the last n dimensions
        if isinstance(dim, int):
            axes = list(range(-dim, 0))
        # If dim is a tuple, restrict along those specific dimensions
        else:
            axes = [a if a < 0 else a - x.ndim for a in dim]

        # Create slices to restrict along each axis
        c = self.n_additional_pts // 2
        slices = [slice(None)] * x.ndim
        for axis in axes:
            slices[axis] = slice(c, -c)

        # Return restricted tensor
        return x[tuple(slices)]
    



class FCLegendre(FourierContinuation):
    """
    Fourier Continuation using Legendre polynomials.
    
    This class implements Fourier Continuation using Legendre polynomial basis functions 
    to extend non-periodic functions to periodic ones on larger domains
    
    Legendre polynomials are orthogonal polynomials with the weight w=1 on the interval [-1, 1].
    """
    
    def __init__(self, d=5, n_additional_pts=50, rcond=1e-15):
        """
        Initialize FCLegendre with specified parameters.
        
        Parameters
        ----------
        d : int
            Number of matching points on the left and right boundaries
            Related to the degree of the Legendre polynomial basis used for extension (degree 2d-1).
            By default 6
        n_additional_pts : int
            Number of additional points to add for continuation, by default 50
            Lower values of d typically require more n_additional_pts points
        rcond : float, optional
            Cutoff for small singular values for the pseudoinverse of the extension matrix
            Singular values less than or equal to rcond * largest_singular_value are set to zero 
            This can be adjusted for numerical stability, especially for larger values of d.
            A smaller value can lead to more stable results, but may also introduce numerical errors.
            By default 1e-15 (same as numpy default for np.linalg.pinv)
        """
        super().__init__(d, n_additional_pts)
        
        self.rcond = rcond
        self.compute_extension_matrix()
    
    
    def compute_extension_matrix(self):
        # Compute the extension matrix using Legendre polynomials.

        total_points = 2*self.d + self.n_additional_pts

        # Use [-1,1] interval where Legendre polynomials are orthogonal
        a, b = -1.0, 1.0
        
        # Generate uniform grid on [-1,1] with total_points points
        h = (b - a) / (total_points - 1)
        full_grid = a + h*np.arange(total_points, dtype=np.float64)
        
        # Extract points for fitting (d points from each end)
        fit_grid = np.concatenate((full_grid[0:self.d], full_grid[-self.d:]), 0)
        
        # Extract points for extension (middle points)
        extension_grid = full_grid[self.d:-self.d]

        # Construct normalized Legendre polynomials
        # numpy.polynomial.legendre.Legendre are orthogonal but not normalized on [-1,1]: 
        #    \int_{-1}^{1} P_j(x) P_k(x) dx = 2/(2j+1) \delta_{jk}
        # Normalization can improve numerical stability for larger values of d
        I = np.eye(2*self.d, dtype=np.float64)
        polynomials = [np.sqrt((2*j+1)/2) * Legendre(I[j, :]) for j in range(2 * self.d)]
        
        # Evaluate normalized polynomials on the fit and extension grids
        X = np.stack([P(fit_grid) for P in polynomials], axis=1)  # Fit grid evaluations
        Q = np.stack([P(extension_grid) for P in polynomials], axis=1)  # Extension grid evaluations

        # Compute extension matrix using pseudoinverse
        ext_mat = Q @ np.linalg.pinv(X, rcond=self.rcond)
        
        # Register matrices as persistent buffers for the module
        self.register_buffer('ext_mat', torch.from_numpy(ext_mat))
        self.register_buffer('ext_mat_T', self.ext_mat.T.clone())

        return self.ext_mat
    

    def _extend_last_axis(self, x):
        """Extend function values along the last axis."""
        
        # Extract boundaries and concatenate them
        right_bnd = x[...,-self.d:]
        left_bnd = x[...,0:self.d]
        y = torch.cat((right_bnd, left_bnd), dim=-1)
        
        # Reshape for matrix multiplication
        y_shape = y.shape
        y = y.reshape(-1, y_shape[-1])
        
        # Apply extension matrix
        ext_mat_T = self.ext_mat_T.to(dtype=x.dtype)
        ext = torch.matmul(y, ext_mat_T + 0j if x.is_complex() else ext_mat_T)
        
        # Reshape back to original shape
        ext = ext.reshape(*y_shape[:-1], ext.shape[-1])
        
        # Concatenate extensions with original signal
        return torch.cat((ext[...,self.n_additional_pts//2:], x, ext[...,:self.n_additional_pts//2]), dim=-1)


class FCGram(FourierContinuation):
    """
    FCGram class for Fourier Continuation using Gram matrices.
    
    This class implements the FC-Gram algorithm described in Section 3.1 of:
    Amlani, F., & Bruno, O. P. (2016). An FC-based spectral solver for
    elastodynamic problems in general three-dimensional domains. 
    Journal of Computational Physics, 307, 333-354.
    
    The algorithm uses pre-computed Gram matrices (ArQr, AlQl) to perform
    Fourier Continuation of discretized functions.
    
    If the required pre-computed FCGram matrices are not available, they 
    can be computed using the MATLAB code from the FCGram repository
    https://github.com/neuraloperator/FCGram/
    and then saved to the appropriate matrices_path directory in .npz format.
    
    The matrices are:
    - ArQr: Right boundary continuation matrix
    - AlQl: Left boundary continuation matrix
    
    The matrices are saved in the .npz format.
    
    """
    
    def __init__(self, d=5, n_additional_pts=50, matrices_path=None):
        """
        Initialize FCGram with specified parameters.
        
        Parameters
        ----------
        d : int
            Number of matching points on the left and right boundaries
            Degree of the Gram polynomiald
            Typically between 3 and 8 (precomputed matrices available for d in {2,3,4,5,6,7,8}).
            d=3,4,5,6 are typically good choices, by default 6.
        n_additional_pts : int
            Number of continuation points. By default 50. 
            Adds n_additional_pts//2 points on each side of the input signal.
            Precomputed matrices available only for n_additional_pts = 50.
            Unlike for FCLegendre, it is usually not necessary to change this parameter. 
        matrices_path : str or Path, optional
            Path to directory containing FCGram matrices. 
            If None, uses the directory containing this file, by default None
        """
        super().__init__(d, n_additional_pts)
        
        if self.n_additional_pts % 2 == 1:
            warnings.warn("n_additional_pts must be even, rounding down.", UserWarning)
            self.n_additional_pts -= 1
        
        self.C = int(self.n_additional_pts // 2)
        
        if matrices_path is None:
            self.matrices_path = Path(__file__).parent / 'fcgram_matrices'
        else:
            self.matrices_path = Path(matrices_path)
        
        # Load pre-computed FCGram matrices
        self.load_matrices()
    
    def load_matrices(self):
        """
        Load the pre-computed FCGram matrices from .npz files.
        
        If the required pre-computed FCGram matrices are not available, they 
        can be computed using the MATLAB code from the FCGram repository
        https://github.com/neuraloperator/FCGram/
        and then saved to the appropriate matrices_path directory in .npz format.
        
        The matrices are:
        - ArQr: Right boundary continuation matrix
        - AlQl: Left boundary continuation matrix
        """
        filepath = self.matrices_path / f'FCGram_data_d{self.d}_C{self.C}.npz'
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"FCGram matrices not found at {filepath}. \n"
                f"Please ensure the .npz file exists with d={self.d}, C={self.C}."
            )
        
        # Load matrices from .npz file
        npz_data = np.load(str(filepath))
        
        # Extract matrices, convert to torch tensors, and register as buffers so they are moved to GPU
        self.register_buffer('ArQr', torch.from_numpy(npz_data['ArQr']))
        self.register_buffer('AlQl', torch.from_numpy(npz_data['AlQl']))
        
    
    def _extend_last_axis(self, x):
        """Extend function values along the last axis using FC-Gram algorithm."""
        
        # Extract boundaries
        left_bnd = x[..., :self.d]      
        right_bnd = x[..., -self.d:]    
        left_shape = left_bnd.shape[:-1]
        right_shape = right_bnd.shape[:-1]
        
        # Convert matrices to correct dtype
        AlQl = self.AlQl.to(dtype=x.dtype)
        ArQr = self.ArQr.to(dtype=x.dtype)
        
        # Reshape for matrix multiplication
        left_bnd = left_bnd.reshape(-1, left_bnd.shape[-1])
        right_bnd = right_bnd.reshape(-1, right_bnd.shape[-1])
        
        # Apply FC-Gram continuation
        left_continuation = torch.matmul(left_bnd, AlQl.T + 0j if x.is_complex() else AlQl.T)
        right_continuation = torch.matmul(right_bnd, ArQr.T + 0j if x.is_complex() else ArQr.T)
        
        # Reshape back to original shape
        left_continuation = left_continuation.reshape(*left_shape, left_continuation.shape[-1])
        right_continuation = right_continuation.reshape(*right_shape, right_continuation.shape[-1])
        
        # Concatenate extensions with original signal
        return torch.cat((left_continuation, x, right_continuation), dim=-1)