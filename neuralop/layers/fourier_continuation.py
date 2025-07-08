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
    """
    
    def __init__(self, d=4, n_additional_pts=50):
        """
        Initialize FCLegendre with specified parameters.
        
        Parameters
        ----------
        d : int
            Number of matching points (degree of approximation), by default 4
        n_additional_pts : int
            Number of additional points to add for continuation, by default 50
        """
        super().__init__()
        
        self.d = d
        self.n_additional_pts = n_additional_pts 
        self.compute_extension_matrix()
    
    def compute_extension_matrix(self):
        # Compute the extension matrix using Legendre polynomials.

        a = 0.0
        h = 0.1

        #Generate grid for the extension
        total_points = 2*self.d + self.n_additional_pts
        full_grid = a + h*np.arange(total_points, dtype=np.float64)
        fit_grid = np.concatenate((full_grid[0:self.d], full_grid[-self.d:]), 0)
        extension_grid = full_grid[self.d:-self.d]

        #Initialize orthogonal polynomial system
        I = np.eye(2*self.d, dtype=np.float64)
        polynomials = []
        for j in range(2*self.d):
            polynomials.append(Legendre(I[j,:], domain=[full_grid[0], full_grid[-1]]))

        #Compute data and evaluation matrices
        X = np.zeros((2*self.d,2*self.d), dtype=np.float64)
        Q = np.zeros((self.n_additional_pts, 2*self.d), dtype=np.float64)
        for j in range(2*self.d):
            Q[:,j] = polynomials[j](extension_grid)
            X[:,j] = polynomials[j](fit_grid)

        #Compute extension matrix
        ext_mat = np.matmul(Q, np.linalg.pinv(X, rcond=1e-31))
        self.register_buffer('ext_mat', torch.from_numpy(ext_mat))
        self.register_buffer('ext_mat_T', self.ext_mat.T.clone())

        return self.ext_mat

    def extend_left_right(self, x, one_sided):
        """
        Extend function values using left-right extension.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., N) where N is the number of points
        one_sided : bool
            If True, extend only to the right. If False, extend to both sides.
            
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
        
        if one_sided:
            return torch.cat((x, ext), dim=-1)
        else:
            return torch.cat((ext[...,self.n_additional_pts//2:], x, ext[...,:self.n_additional_pts//2]), dim=-1)
    
    
    def extend_top_bottom(self, x, one_sided):
        """
        Extend function values using top-bottom extension.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., M, N) where M, N are spatial dimensions
        one_sided : bool
            If True, extend only to the bottom. If False, extend to both sides.
            
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
        
        if one_sided:
            return torch.cat((x, ext), dim=-2)
        else:
            return torch.cat((ext[...,self.n_additional_pts//2:,:], x, ext[...,:self.n_additional_pts//2,:]), dim=-2)

    def extend_front_back(self, x, one_sided):
        """
        Extend function values using front-back extension.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., L, M, N) where L, M, N are spatial dimensions
        one_sided : bool
            If True, extend only to the back. If False, extend to both sides.
            
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

        if one_sided:
            return torch.cat((x, ext), dim=-3)
        else:
            return torch.cat((ext[..., self.n_additional_pts//2:, :, :], x, ext[..., :self.n_additional_pts//2, :, :]), dim=-3)

    def extend1d(self, x, one_sided):
        """
        Extend 1D function values.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., N)
        one_sided : bool
            If True, extend only to the right. If False, extend to both sides.
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        return self.extend_left_right(x, one_sided)
    
    def extend2d(self, x, one_sided):
        """
        Extend 2D function values.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., M, N)
        one_sided : bool
            If True, extend only to one side. If False, extend to both sides.
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        x = self.extend_left_right(x, one_sided)
        x = self.extend_top_bottom(x, one_sided)
        return x

    def extend3d(self, x, one_sided):
        """
        Extend 3D function values.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., L, M, N)
        one_sided : bool
            If True, extend only to one side. If False, extend to both sides.
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        x = self.extend_left_right(x, one_sided)
        x = self.extend_top_bottom(x, one_sided)
        x = self.extend_front_back(x, one_sided)
        return x
    
    def forward(self, x, dim=2, one_sided=True):
        """
        Forward pass for Fourier continuation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to extend
        dim : int, optional
            Dimension of the problem (1, 2, or 3), by default 2
        one_sided : bool, optional
            If True, extend only to one side. If False, extend to both sides, by default True
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        if dim == 1:
            return self.extend1d(x, one_sided)
        if dim == 2:
            return self.extend2d(x, one_sided)
        if dim == 3:
            return self.extend3d(x, one_sided)




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
    
    def __init__(self, d=4, n_additional_pts=50, matrices_path=None):
        """
        Initialize FCGram with specified parameters.
        
        Parameters
        ----------
        d : int
            Number of matching points, typically between 2 and 8. 
            d=3,4,5,6 are typically good choices, by default 4.
        n_additional_pts : int
            Number of continuation points. By default 50.
            Unlike for FCLegendre, it is usually not necessary to change this parameter. 
        matrices_path : str or Path, optional
            Path to directory containing FCGram matrices. 
            If None, uses the directory containing this file, by default None
        """
        super().__init__()
        
        self.d = d
        if n_additional_pts % 2 == 1:
            warnings.warn("n_additional_pts must be even, rounding down.", UserWarning)
            n_additional_pts -= 1
        self.C = int(n_additional_pts // 2)
        
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
        
    
    def extend_left_right(self, x, one_sided=True):
        """
        Extend function values using FC-Gram algorithm for left-right extension.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., N) where N is the number of points
        one_sided : bool, optional
            If True, extend only to the right. If False, extend to both sides, by default True
            
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
        
        if one_sided:
            return torch.cat((x, right_continuation, left_continuation), dim=-1)
        else:
            return torch.cat((left_continuation, x, right_continuation), dim=-1)
    
    def extend_top_bottom(self, x, one_sided=True):
        """
        Extend function values using FC-Gram algorithm for top-bottom extension.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., M, N)
        one_sided : bool, optional
            If True, extend only to the bottom. If False, extend to both sides, by default True
            
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
        
        if one_sided:
            return torch.cat((x, bottom_continuation, top_continuation), dim=-2)
        else:
            return torch.cat((top_continuation, x, bottom_continuation), dim=-2)
    
    
    def extend_front_back(self, x, one_sided=True):
        """
        Extend function values using FC-Gram algorithm for front-back extension.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., D, M, N)
        one_sided : bool, optional
            If True, extend only to the back. If False, extend to both sides, by default True
            
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
        
        if one_sided:
            return torch.cat((x, back_continuation, front_continuation), dim=-3)
        else:
            return torch.cat((front_continuation, x, back_continuation), dim=-3)

    def extend1d(self, x, one_sided=True):
        """
        Extend 1D function values using FC-Gram algorithm.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., N)
        one_sided : bool, optional
            If True, extend only to the right. If False, extend to both sides, by default True
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        return self.extend_left_right(x, one_sided)
    
    def extend2d(self, x, one_sided=True):
        """
        Extend 2D function values using FC-Gram algorithm.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., M, N)
        one_sided : bool, optional
            If True, extend only to one side. If False, extend to both sides, by default True
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        x = self.extend_left_right(x, one_sided)
        x = self.extend_top_bottom(x, one_sided)
        return x

    def extend3d(self, x, one_sided=True):
        """
        Extend 3D function values using FC-Gram algorithm.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., L, M, N)
        one_sided : bool, optional
            If True, extend only to one side. If False, extend to both sides, by default True
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        x = self.extend_left_right(x, one_sided)
        x = self.extend_top_bottom(x, one_sided)
        x = self.extend_front_back(x, one_sided)
        return x
    
    def forward(self, x, dim=2, one_sided=True):
        """
        Forward pass for FC-Gram Fourier continuation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to extend
        dim : int, optional
            Dimension of the problem (1, 2, or 3), by default 2
        one_sided : bool, optional
            If True, extend only to one side. If False, extend to both sides, by default True
            
        Returns
        -------
        torch.Tensor
            Extended function values
        """
        if dim == 1:
            return self.extend1d(x, one_sided)
        if dim == 2:
            return self.extend2d(x, one_sided)
        if dim == 3:
            return self.extend3d(x, one_sided)