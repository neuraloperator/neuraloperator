import torch
import torch.nn as nn
import numpy as np
from numpy.polynomial.legendre import Legendre
import warnings
from pathlib import Path


class FCLegendre(nn.Module):
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
        super().__init__()
        
        self.d = d
        self.n_additional_pts = n_additional_pts 
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
    

    def extend_left_right(self, x):
        """
        Extend function values using left-right extension.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., N) where N is the number of points
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        right_bnd = x[...,-self.d:]
        left_bnd = x[...,0:self.d]
        y = torch.cat((right_bnd, left_bnd), dim=-1)
        
        ext_mat_T = self.ext_mat_T.to(dtype=x.dtype)
        if x.is_complex():
            ext = torch.matmul(y, ext_mat_T + 0j)
        else:
            ext = torch.matmul(y, ext_mat_T)
        
        return torch.cat((ext[...,self.n_additional_pts//2:], x, ext[...,:self.n_additional_pts//2]), dim=-1)
    
    
    def extend_top_bottom(self, x):
        """
        Extend function values using top-bottom extension.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., M, N) where M, N are spatial dimensions
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        bottom_bnd = x[...,-self.d:,:]
        top_bnd = x[...,0:self.d,:]
        y = torch.cat((bottom_bnd, top_bnd), dim=-2)
        
        ext_mat = self.ext_mat.to(dtype=x.dtype)
        if x.is_complex():
            ext = torch.matmul(ext_mat, y + 0j)
        else:
            ext = torch.matmul(ext_mat, y)
        
        return torch.cat((ext[...,self.n_additional_pts//2:,:], x, ext[...,:self.n_additional_pts//2,:]), dim=-2)

    def extend_front_back(self, x):
        """
        Extend function values using front-back extension.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., L, M, N) where L, M, N are spatial dimensions
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        back_bnd = x[..., -self.d:, :, :]
        front_bnd = x[..., :self.d, :, :]
        y = torch.cat((back_bnd, front_bnd), dim=-3)

        y_shape = y.shape
        y_reshaped = y.reshape(*y_shape[:-3], y_shape[-3], -1)

        ext_mat = self.ext_mat.to(dtype=x.dtype)
        if x.is_complex():
            ext_reshaped = torch.matmul(ext_mat, y_reshaped + 0j)
        else:
            ext_reshaped = torch.matmul(ext_mat, y_reshaped)

        ext = ext_reshaped.reshape(*y_shape[:-3], self.n_additional_pts, y_shape[-2], y_shape[-1])

        return torch.cat((ext[..., self.n_additional_pts//2:, :, :], x, ext[..., :self.n_additional_pts//2, :, :]), dim=-3)

    def extend1d(self, x):
        """
        Extend 1D function values.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., N)
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        return self.extend_left_right(x)
    
    def extend2d(self, x):
        """
        Extend 2D function values.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., M, N)
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        x = self.extend_left_right(x)
        x = self.extend_top_bottom(x)
        return x

    def extend3d(self, x):
        """
        Extend 3D function values.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., L, M, N)
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        x = self.extend_left_right(x)
        x = self.extend_top_bottom(x)
        x = self.extend_front_back(x)
        return x
    
    def forward(self, x, dim=2):
        """
        Forward pass for Fourier continuation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to extend
        dim : int, optional
            Dimension of the problem (1, 2, or 3), by default 2
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        if dim == 1:
            return self.extend1d(x)
        if dim == 2:
            return self.extend2d(x)
        if dim == 3:
            return self.extend3d(x)

    def restrict(self, x, dim):
        """
        Remove Fourier continuation extension points to restore original domain size.
        
        Reverses the extension process by removing half of the additional points
        on each side that were added during Fourier continuation.
        
        Parameters
        ----------
        x : torch.Tensor
            Extended tensor from Fourier continuation
        dim : int
            Number of dimensions to restrict (1, 2, or 3)
        
        Returns
        -------
        torch.Tensor
            Tensor with original domain size, half of extension points removed from each side
        """
        c = self.n_additional_pts // 2
        return x[(Ellipsis,) + (slice(c, -c),) * dim]




class FCGram(nn.Module):
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
        super().__init__()
        
        self.d = d
        self.n_additional_pts = n_additional_pts 
        
        if self.n_additional_pts % 2 == 1:
            warnings.warn("n_additional_pts must be even, rounding down.", UserWarning)
            self.n_additional_pts -= 1
        self.C = int(self.n_additional_pts // 2)
        
        if matrices_path is None:
            self.matrices_path = Path(__file__).parent / 'FCGram_matrices'
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
        
    
    def extend_left_right(self, x):
        """
        Extend function values using FC-Gram algorithm for left-right extension.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., N) where N is the number of points
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        # Extract boundary values for continuation, use d points from each boundary
        left_bnd = x[..., :self.d]      
        right_bnd = x[..., -self.d:]    
        
        AlQl = self.AlQl.to(dtype=x.dtype)
        ArQr = self.ArQr.to(dtype=x.dtype)
        
        # Apply FC-Gram continuation using ArQr matrix
        if x.is_complex():
            left_continuation = torch.matmul(left_bnd, AlQl.T + 0j)
            right_continuation = torch.matmul(right_bnd, ArQr.T + 0j)
        else:
            left_continuation = torch.matmul(left_bnd, AlQl.T)
            right_continuation = torch.matmul(right_bnd, ArQr.T)
        
        return torch.cat((left_continuation, x, right_continuation), dim=-1)
    
    def extend_top_bottom(self, x):
        """
        Extend function values using FC-Gram algorithm for top-bottom extension.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., M, N)
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        # Extract boundary values for continuation
        top_bnd = x[..., :self.d, :]     
        bottom_bnd = x[..., -self.d:, :] 
        
        AlQl = self.AlQl.to(dtype=x.dtype)
        ArQr = self.ArQr.to(dtype=x.dtype)
        
        # Apply FC-Gram continuation using ArQr matrix
        if x.is_complex():
            bottom_continuation = torch.matmul(ArQr, bottom_bnd + 0j)
            top_continuation = torch.matmul(AlQl, top_bnd + 0j)
        else:
            bottom_continuation = torch.matmul(ArQr, bottom_bnd)
            top_continuation = torch.matmul(AlQl, top_bnd)
        
        return torch.cat((top_continuation, x, bottom_continuation), dim=-2)
    
    
    def extend_front_back(self, x):
        """
        Extend function values using FC-Gram algorithm for front-back extension.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., D, M, N)
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        # Extract boundary values for continuation
        front_bnd = x[..., :self.d, :, :]     
        back_bnd = x[..., -self.d:, :, :] 
        
        AlQl = self.AlQl.to(dtype=x.dtype)
        ArQr = self.ArQr.to(dtype=x.dtype)
        
        # Reshape for matrix multiplication
        y_shape = x.shape
        front_bnd_reshaped = front_bnd.reshape(*y_shape[:-3], self.d, -1)
        back_bnd_reshaped = back_bnd.reshape(*y_shape[:-3], self.d, -1)
        
        # Apply FC-Gram continuation
        if x.is_complex():
            front_continuation_reshaped = torch.matmul(AlQl, front_bnd_reshaped + 0j)
            back_continuation_reshaped = torch.matmul(ArQr, back_bnd_reshaped + 0j)
        else:
            front_continuation_reshaped = torch.matmul(AlQl, front_bnd_reshaped)
            back_continuation_reshaped = torch.matmul(ArQr, back_bnd_reshaped)
        
        # Reshape back to original dimensions
        front_continuation = front_continuation_reshaped.reshape(*y_shape[:-3], self.C, y_shape[-2], y_shape[-1])
        back_continuation = back_continuation_reshaped.reshape(*y_shape[:-3], self.C, y_shape[-2], y_shape[-1])
        
        return torch.cat((front_continuation, x, back_continuation), dim=-3)

    def extend1d(self, x):
        """
        Extend 1D function values using FC-Gram algorithm.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., N)
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        return self.extend_left_right(x)
    
    def extend2d(self, x):
        """
        Extend 2D function values using FC-Gram algorithm.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., M, N)
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        x = self.extend_left_right(x)
        x = self.extend_top_bottom(x)
        return x

    def extend3d(self, x):
        """
        Extend 3D function values using FC-Gram algorithm.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., L, M, N)
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        x = self.extend_left_right(x)
        x = self.extend_top_bottom(x)
        x = self.extend_front_back(x)
        return x
    
    def forward(self, x, dim=2):
        """
        Forward pass for FC-Gram Fourier continuation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to extend
        dim : int, optional
            Dimension of the problem (1, 2, or 3), by default 2
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        if dim == 1:
            return self.extend1d(x)
        if dim == 2:
            return self.extend2d(x)
        if dim == 3:
            return self.extend3d(x)

    def restrict(self, x, dim):
        """
        Remove Fourier continuation extension points to restore original domain size.
        
        Reverses the extension process by removing half of the additional points
        on each side that were added during Fourier continuation.
        
        Parameters
        ----------
        x : torch.Tensor
            Extended tensor from Fourier continuation
        dim : int
            Number of dimensions to restrict (1, 2, or 3)
        
        Returns
        -------
        torch.Tensor
            Tensor with original domain size, half of extension points removed from each side
        """
        c = self.n_additional_pts // 2
        return x[(Ellipsis,) + (slice(c, -c),) * dim]