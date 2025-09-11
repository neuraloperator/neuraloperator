import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Union, Literal
import math

from functools import partial
from scipy.special import eval_legendre
from sympy import Poly, legendre, Symbol, chebyshevt


class WaveletUtils:
    """
    Unified wavelet utility class containing wavelet transform functions for all dimensions
    """

    @staticmethod
    def legendreDer(k, x):
        """
        Compute the derivative of Legendre polynomial using the recurrence relation.
        
        The derivative of P_k(x) can be expressed as a sum of lower-order Legendre polynomials:
        P'_k(x) = (2k+1)P_{k-1}(x) + (2(k-2)+1)P_{k-3}(x) + ... 
        
        Parameters
        ----------
        k : int
            Order of the Legendre polynomial
        x : array_like
            Points at which to evaluate the derivative
            
        Returns
        -------
        out : array_like
            Derivative values at the given points
        """

        def _legendre(k, x):
            return (2*k+1) * eval_legendre(k, x)
        out = 0
        for i in np.arange(k-1,-1,-2):
            out += _legendre(i, x)
        return out

    @staticmethod
    def phi_(phi_c, x, lb=0, ub=1):
        """
        Evaluate a polynomial basis function with compact support.
        
        This function evaluates a polynomial defined by coefficients phi_c,
        but sets the value to zero outside the interval [lb, ub].
        
        Parameters
        ----------
        phi_c : array_like
            Polynomial coefficients
        x : array_like
            Points at which to evaluate the polynomial
        lb : float, optional
            Lower bound of the support interval, default 0
        ub : float, optional
            Upper bound of the support interval, default 1
            
        Returns
        -------
        array_like
            Polynomial values, zero outside [lb, ub]
        """
        mask = np.logical_or(x<lb, x>ub) * 1.0
        return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1-mask)

    @classmethod
    def get_phi_psi(cls, k, base):
        """
        Generate scaling functions (phi) and wavelet functions (psi) for the given basis.
        
        This method constructs orthogonal scaling and wavelet functions using either
        Legendre or Chebyshev polynomials as the base. The construction follows a
        Gram-Schmidt-like orthogonalization procedure.
        
        Parameters
        ----------
        k : int
            Number of basis functions (determines the approximation order)
        base : str
            Type of polynomial basis ('legendre' or 'chebyshev')
            
        Returns
        -------
        phi : list of functions
            List of k scaling functions
        psi1 : list of functions
            List of k wavelet functions for the first half of the interval
        psi2 : list of functions
            List of k wavelet functions for the second half of the interval
            
        Notes
        -----
        For Legendre basis:
        - Scaling functions are normalized Legendre polynomials on [0,1]
        - Wavelets are constructed via orthogonalization against scaling functions
        
        For Chebyshev basis:
        - Scaling functions are normalized Chebyshev polynomials
        - Wavelets have compact support on [0,0.5] and [0.5,1] respectively
        """
        x = Symbol('x')
        phi_coeff = np.zeros((k,k))
        phi_2x_coeff = np.zeros((k,k))
        
        if base == 'legendre':
            for ki in range(k):
                coeff_ = Poly(legendre(ki, 2*x-1), x).all_coeffs()
                phi_coeff[ki,:ki+1] = np.flip(np.sqrt(2*ki+1) * np.array(coeff_).astype(np.float64))
                coeff_ = Poly(legendre(ki, 4*x-1), x).all_coeffs()
                phi_2x_coeff[ki,:ki+1] = np.flip(np.sqrt(2) * np.sqrt(2*ki+1) * np.array(coeff_).astype(np.float64))
            
            psi1_coeff = np.zeros((k, k))
            psi2_coeff = np.zeros((k, k))
            for ki in range(k):
                psi1_coeff[ki,:] = phi_2x_coeff[ki,:]
                for i in range(k):
                    a = phi_2x_coeff[ki,:ki+1]
                    b = phi_coeff[i, :i+1]
                    prod_ = np.convolve(a, b)
                    prod_[np.abs(prod_)<1e-8] = 0
                    proj_ = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, 1+np.arange(len(prod_)))).sum()
                    psi1_coeff[ki,:] -= proj_ * phi_coeff[i,:]
                    psi2_coeff[ki,:] -= proj_ * phi_coeff[i,:]
                    
                for j in range(ki):
                    a = phi_2x_coeff[ki,:ki+1]
                    b = psi1_coeff[j, :]
                    prod_ = np.convolve(a, b)
                    prod_[np.abs(prod_)<1e-8] = 0
                    proj_ = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, 1+np.arange(len(prod_)))).sum()
                    psi1_coeff[ki,:] -= proj_ * psi1_coeff[j,:]
                    psi2_coeff[ki,:] -= proj_ * psi2_coeff[j,:]

                a = psi1_coeff[ki,:]
                prod_ = np.convolve(a, a)
                prod_[np.abs(prod_)<1e-8] = 0
                norm1 = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, 1+np.arange(len(prod_)))).sum()

                a = psi2_coeff[ki,:]
                prod_ = np.convolve(a, a)
                prod_[np.abs(prod_)<1e-8] = 0
                norm2 = (prod_ * 1/(np.arange(len(prod_))+1) * (1-np.power(0.5, 1+np.arange(len(prod_))))).sum()
                norm_ = np.sqrt(norm1 + norm2)
                psi1_coeff[ki,:] /= norm_
                psi2_coeff[ki,:] /= norm_
                psi1_coeff[np.abs(psi1_coeff)<1e-8] = 0
                psi2_coeff[np.abs(psi2_coeff)<1e-8] = 0

            phi = [np.poly1d(np.flip(phi_coeff[i,:])) for i in range(k)]
            psi1 = [np.poly1d(np.flip(psi1_coeff[i,:])) for i in range(k)]
            psi2 = [np.poly1d(np.flip(psi2_coeff[i,:])) for i in range(k)]
            
        elif base == 'chebyshev':
            for ki in range(k):
                if ki == 0:
                    phi_coeff[ki,:ki+1] = np.sqrt(2/np.pi)
                    phi_2x_coeff[ki,:ki+1] = np.sqrt(2/np.pi) * np.sqrt(2)
                else:
                    coeff_ = Poly(chebyshevt(ki, 2*x-1), x).all_coeffs()
                    phi_coeff[ki,:ki+1] = np.flip(2/np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
                    coeff_ = Poly(chebyshevt(ki, 4*x-1), x).all_coeffs()
                    phi_2x_coeff[ki,:ki+1] = np.flip(np.sqrt(2) * 2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
            
            phi = [partial(cls.phi_, phi_coeff[i,:]) for i in range(k)]
            psi1 = [partial(cls.phi_, np.zeros(k), lb=0, ub=0.5) for i in range(k)]
            psi2 = [partial(cls.phi_, np.zeros(k), lb=0.5, ub=1) for i in range(k)]
        
        return phi, psi1, psi2

    @classmethod
    def get_filter(cls, base, k):
        """
        Generate filter bank coefficients for wavelet decomposition and reconstruction.
        
        This method computes the two-scale relation coefficients that relate scaling
        and wavelet functions at different scales. These filters are essential for
        the fast wavelet transform algorithm.
        
        Parameters
        ----------
        base : str
            Type of polynomial basis ('legendre' or 'chebyshev')
        k : int
            Number of basis functions
            
        Returns
        -------
        H0 : ndarray of shape (k, k)
            Low-pass filter for even samples (scaling coefficients)
        H1 : ndarray of shape (k, k)
            Low-pass filter for odd samples (scaling coefficients)
        G0 : ndarray of shape (k, k)
            High-pass filter for even samples (wavelet coefficients)
        G1 : ndarray of shape (k, k)
            High-pass filter for odd samples (wavelet coefficients)
        PHI0 : ndarray of shape (k, k)
            Reconstruction matrix for even samples
        PHI1 : ndarray of shape (k, k)
            Reconstruction matrix for odd samples
            
        Notes
        -----
        The filters satisfy the two-scale relations:
        - φ(x) = √2 * Σ H[n] * φ(2x - n)
        - ψ(x) = √2 * Σ G[n] * φ(2x - n)
        
        Where H are low-pass (scaling) filters and G are high-pass (wavelet) filters.
        """
        def psi(psi1, psi2, i, inp):
            mask = (inp<=0.5) * 1.0
            return psi1[i](inp) * mask + psi2[i](inp) * (1-mask)
        
        if base not in ['legendre', 'chebyshev']:
            raise Exception('Base not supported')
        
        x = Symbol('x')
        H0 = np.zeros((k,k))
        H1 = np.zeros((k,k))
        G0 = np.zeros((k,k))
        G1 = np.zeros((k,k))
        PHI0 = np.zeros((k,k))
        PHI1 = np.zeros((k,k))
        
        phi, psi1, psi2 = cls.get_phi_psi(k, base)
        
        if base == 'legendre':
            roots = Poly(legendre(k, 2*x-1)).all_roots()
            x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
            wm = 1/k/cls.legendreDer(k,2*x_m-1)/eval_legendre(k-1,2*x_m-1)
            
            for ki in range(k):
                for kpi in range(k):
                    H0[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki](x_m/2) * phi[kpi](x_m)).sum()
                    G0[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m/2) * phi[kpi](x_m)).sum()
                    H1[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki]((x_m+1)/2) * phi[kpi](x_m)).sum()
                    G1[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m+1)/2) * phi[kpi](x_m)).sum()
                    
            PHI0 = np.eye(k)
            PHI1 = np.eye(k)
            
        elif base == 'chebyshev':
            kUse = 2*k
            roots = Poly(chebyshevt(kUse, 2*x-1)).all_roots()
            x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
            wm = np.pi / kUse / 2

            for ki in range(k):
                for kpi in range(k):
                    H0[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki](x_m/2) * phi[kpi](x_m)).sum()
                    G0[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m/2) * phi[kpi](x_m)).sum()
                    H1[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki]((x_m+1)/2) * phi[kpi](x_m)).sum()
                    G1[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m+1)/2) * phi[kpi](x_m)).sum()

                    PHI0[ki, kpi] = (wm * phi[ki](2*x_m) * phi[kpi](2*x_m)).sum() * 2
                    PHI1[ki, kpi] = (wm * phi[ki](2*x_m-1) * phi[kpi](2*x_m-1)).sum() * 2
                    
            PHI0[np.abs(PHI0)<1e-8] = 0
            PHI1[np.abs(PHI1)<1e-8] = 0

        for matrix in [H0, H1, G0, G1]:
            matrix[np.abs(matrix)<1e-8] = 0
            
        return H0, H1, G0, G1, PHI0, PHI1




class SparseKernel(nn.Module):
    """
    Dimension-agnostic sparse kernel layer
    """

    def __init__(self, k, alpha, c=1, n_dim=1, **kwargs):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.c = c
        self.n_dim = n_dim
        
        if n_dim == 1:
            in_channels = c * k
        elif n_dim in [2]:
            in_channels = c * k**2
        else:
            raise ValueError(f"Unsupported dimension: {n_dim}")
        
        self.conv = self._build_conv_block(in_channels, 128)
        self.output_proj = nn.Linear(128, in_channels)
    
    def _build_conv_block(self, in_channels, out_channels):
        # Build appropriate convolution block based on dimension
        if self.n_dim == 1:
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 3, 1, 1),
                nn.ReLU(inplace=True)
            )
        elif self.n_dim == 2:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        original_shape = x.shape
        
        if self.n_dim == 1:
            B, N, c, k = x.shape
            x = x.view(B, N, -1).permute(0, 2, 1)
        elif self.n_dim == 2:
            B, Nx, Ny, c, k_sq = x.shape
            x = x.view(B, Nx, Ny, -1).permute(0, 3, 1, 2)
        
        x = self.conv(x)
        
        if self.n_dim == 1:
            x = x.permute(0, 2, 1)
        elif self.n_dim == 2:
            x = x.permute(0, 2, 3, 1)
        
        x = self.output_proj(x)
        x = x.view(original_shape)
        return x


class SparseKernelFT(nn.Module):
    """
    Dimension-agnostic Fourier sparse kernel layer
    """

    def __init__(self, k, alpha, c=1, n_dim=1, **kwargs):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.c = c
        self.n_dim = n_dim
        
        if n_dim == 1:
            in_channels = c * k
            self.weights = nn.Parameter(
                torch.rand(in_channels, in_channels, alpha, dtype=torch.cfloat) / (in_channels * in_channels)
            )
        elif n_dim == 2:
            in_channels = c * k**2
            self.weights1 = nn.Parameter(torch.zeros(in_channels, in_channels, alpha, alpha, dtype=torch.cfloat))
            self.weights2 = nn.Parameter(torch.zeros(in_channels, in_channels, alpha, alpha, dtype=torch.cfloat))
            nn.init.xavier_normal_(self.weights1)
            nn.init.xavier_normal_(self.weights2)
        
        if n_dim in [2]:
            self.output_proj = nn.Linear(in_channels, in_channels)
    
    def forward(self, x):
        original_shape = x.shape
        
        if self.n_dim == 1:
            B, N, c, k = x.shape
            x = x.view(B, N, -1).permute(0, 2, 1)
            x_fft = torch.fft.rfft(x)
            
            l = min(self.alpha, N//2+1)
            out_ft = torch.zeros(B, c*k, N//2 + 1, device=x.device, dtype=torch.cfloat)
            out_ft[:, :, :l] = torch.einsum("bix,iox->box", x_fft[:, :, :l], self.weights[:, :, :l])
            
            x = torch.fft.irfft(out_ft, n=N)
            x = x.permute(0, 2, 1).view(original_shape)
            
        elif self.n_dim == 2:
            B, Nx, Ny, c, k_sq = x.shape
            x = x.view(B, Nx, Ny, -1).permute(0, 3, 1, 2)
            x_fft = torch.fft.rfft2(x)
            
            l1 = min(self.alpha, Nx//2+1)
            l2 = min(self.alpha, Ny//2+1)
            out_ft = torch.zeros(B, c*k_sq, Nx, Ny//2 + 1, device=x.device, dtype=torch.cfloat)
            
            out_ft[:, :, :l1, :l2] = torch.einsum("bixy,ioxy->boxy", 
                                                  x_fft[:, :, :l1, :l2], self.weights1[:, :, :l1, :l2])
            out_ft[:, :, -l1:, :l2] = torch.einsum("bixy,ioxy->boxy", 
                                                   x_fft[:, :, -l1:, :l2], self.weights2[:, :, :l1, :l2])
            
            x = torch.fft.irfft2(out_ft, s=(Nx, Ny))
            x = x.permute(0, 2, 3, 1)
            x = F.relu(x)
            x = self.output_proj(x)
            x = x.view(original_shape)
            
        return x



class MWNO_CZ(nn.Module):
    """
    Dimension-agnostic MWNO core layer
    """

    def __init__(self, k=3, alpha=5, L=0, c=1, base='legendre', n_dim=1, initializer=None, **kwargs):
        super().__init__()
        
        self.k = k
        self.L = L
        self.n_dim = n_dim
        
        H0, H1, G0, G1, PHI0, PHI1 = WaveletUtils.get_filter(base, k)
        H0r = H0@PHI0
        G0r = G0@PHI0
        H1r = H1@PHI1
        G1r = G1@PHI1
        
        for matrix in [H0r, H1r, G0r, G1r]:
            matrix[np.abs(matrix)<1e-8] = 0
        
        self.A = SparseKernelFT(k, alpha, c, n_dim)
        self.B = SparseKernelFT(k, alpha, c, n_dim) if n_dim == 1 else SparseKernel(k, alpha, c, n_dim)
        self.C = SparseKernel(k, alpha, c, n_dim)
        
        if n_dim == 1:
            self.T0 = nn.Linear(k, k)
        else:
            self.T0 = nn.Linear(c*k**2, c*k**2)
        
        if initializer is not None:
            initializer(self.T0.weight)
        
        self._register_wavelet_filters(H0, H1, G0, G1, H0r, H1r, G0r, G1r)
    
    def _register_wavelet_filters(self, H0, H1, G0, G1, H0r, H1r, G0r, G1r):
        # Register wavelet filters
        if self.n_dim == 1:
            self.register_buffer('ec_s', torch.Tensor(np.concatenate((H0.T, H1.T), axis=0)))
            self.register_buffer('ec_d', torch.Tensor(np.concatenate((G0.T, G1.T), axis=0)))
            self.register_buffer('rc_e', torch.Tensor(np.concatenate((H0r, G0r), axis=0)))
            self.register_buffer('rc_o', torch.Tensor(np.concatenate((H1r, G1r), axis=0)))
            
        elif self.n_dim >= 2:
            self.register_buffer('ec_s', torch.Tensor(
                np.concatenate((np.kron(H0, H0).T, np.kron(H0, H1).T,
                               np.kron(H1, H0).T, np.kron(H1, H1).T), axis=0)))
            self.register_buffer('ec_d', torch.Tensor(
                np.concatenate((np.kron(G0, G0).T, np.kron(G0, G1).T,
                               np.kron(G1, G0).T, np.kron(G1, G1).T), axis=0)))
            
            self.register_buffer('rc_ee', torch.Tensor(np.concatenate((np.kron(H0r, H0r), np.kron(G0r, G0r)), axis=0)))
            self.register_buffer('rc_eo', torch.Tensor(np.concatenate((np.kron(H0r, H1r), np.kron(G0r, G1r)), axis=0)))
            self.register_buffer('rc_oe', torch.Tensor(np.concatenate((np.kron(H1r, H0r), np.kron(G1r, G0r)), axis=0)))
            self.register_buffer('rc_oo', torch.Tensor(np.concatenate((np.kron(H1r, H1r), np.kron(G1r, G1r)), axis=0)))
    
    def wavelet_transform(self, x):
        # Dimension-agnostic wavelet transform
        if self.n_dim == 1:
            xa = torch.cat([x[:, ::2, :, :], x[:, 1::2, :, :]], -1)
        elif self.n_dim >= 2:
            xa = torch.cat([x[..., ::2, ::2, :, :], x[..., ::2, 1::2, :, :], 
                           x[..., 1::2, ::2, :, :], x[..., 1::2, 1::2, :, :]], -1)
        
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s
    
    def even_odd_reconstruction(self, x):
        # Dimension-agnostic even-odd reconstruction
        if self.n_dim == 1:
            B, N, c, ich = x.shape
            assert ich == 2*self.k
            x_e = torch.matmul(x, self.rc_e)
            x_o = torch.matmul(x, self.rc_o)
            
            result = torch.zeros(B, N*2, c, self.k, device=x.device)
            result[..., ::2, :, :] = x_e
            result[..., 1::2, :, :] = x_o
            return result
            
        elif self.n_dim == 2:
            B, Nx, Ny, c, ich = x.shape
            assert ich == 2*self.k**2
            x_ee = torch.matmul(x, self.rc_ee)
            x_eo = torch.matmul(x, self.rc_eo)
            x_oe = torch.matmul(x, self.rc_oe)
            x_oo = torch.matmul(x, self.rc_oo)
            
            result = torch.zeros(B, Nx*2, Ny*2, c, self.k**2, device=x.device)
            result[:, ::2, ::2, :, :] = x_ee
            result[:, ::2, 1::2, :, :] = x_eo
            result[:, 1::2, ::2, :, :] = x_oe
            result[:, 1::2, 1::2, :, :] = x_oo
            return result
            
    
    def forward(self, x):
        if self.n_dim == 1:
            B, N, c, ich = x.shape
            ns = math.floor(np.log2(N))
        else:
            B, *spatial_dims, c, ich = x.shape
            ns = math.floor(np.log2(spatial_dims[0]))
        
        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])
        
        for i in range(ns - self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        
        if self.n_dim == 1:
            x = self.T0(x)
        else:
            coarse_size = 2**self.L
            if self.n_dim == 2:
                x = self.T0(x.view(B, coarse_size, coarse_size, -1)).view(B, coarse_size, coarse_size, c, ich)
        
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.even_odd_reconstruction(x)
        
        return x