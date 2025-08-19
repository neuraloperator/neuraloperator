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
    """Unified wavelet utility class containing wavelet transform functions for all dimensions"""
    
    @staticmethod
    def legendreDer(k, x):
        def _legendre(k, x):
            return (2*k+1) * eval_legendre(k, x)
        out = 0
        for i in np.arange(k-1,-1,-2):
            out += _legendre(i, x)
        return out

    @staticmethod
    def phi_(phi_c, x, lb=0, ub=1):
        mask = np.logical_or(x<lb, x>ub) * 1.0
        return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1-mask)

    @classmethod
    def get_phi_psi(cls, k, base):
        """Get wavelet basis functions"""
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
        """Get wavelet filters"""
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
    """Dimension-agnostic sparse kernel layer"""
    
    def __init__(self, k, alpha, c=1, n_dim=1, **kwargs):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.c = c
        self.n_dim = n_dim
        
        if n_dim == 1:
            in_channels = c * k
        elif n_dim in [2, 3]:
            in_channels = c * k**2
        else:
            raise ValueError(f"Unsupported dimension: {n_dim}")
        
        self.conv = self._build_conv_block(in_channels, 128)
        self.output_proj = nn.Linear(128, in_channels)
    
    def _build_conv_block(self, in_channels, out_channels):
        """Build appropriate convolution block based on dimension"""
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
        elif self.n_dim == 3:
            return nn.Sequential(
                nn.Conv3d(out_channels, out_channels, 3, 1, 1),
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
        elif self.n_dim == 3:
            B, Nx, Ny, T, c, k_sq = x.shape
            x = x.view(B, Nx, Ny, T, -1).permute(0, 4, 1, 2, 3)
        
        x = self.conv(x)
        
        if self.n_dim == 1:
            x = x.permute(0, 2, 1)
        elif self.n_dim == 2:
            x = x.permute(0, 2, 3, 1)
        elif self.n_dim == 3:
            x = x.permute(0, 2, 3, 4, 1)
        
        x = self.output_proj(x)
        x = x.view(original_shape)
        return x


class SparseKernelFT(nn.Module):
    """Dimension-agnostic Fourier sparse kernel layer"""
    
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
        elif n_dim == 3:
            in_channels = c * k**2
            self.weights1 = nn.Parameter(torch.zeros(in_channels, in_channels, alpha, alpha, alpha, dtype=torch.cfloat))
            self.weights2 = nn.Parameter(torch.zeros(in_channels, in_channels, alpha, alpha, alpha, dtype=torch.cfloat))
            self.weights3 = nn.Parameter(torch.zeros(in_channels, in_channels, alpha, alpha, alpha, dtype=torch.cfloat))
            self.weights4 = nn.Parameter(torch.zeros(in_channels, in_channels, alpha, alpha, alpha, dtype=torch.cfloat))
            for w in [self.weights1, self.weights2, self.weights3, self.weights4]:
                nn.init.xavier_normal_(w)
        
        if n_dim in [2, 3]:
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
            
        elif self.n_dim == 3:
            B, Nx, Ny, T, c, k_sq = x.shape
            x = x.view(B, Nx, Ny, T, -1).permute(0, 4, 1, 2, 3)
            x_fft = torch.fft.rfftn(x, dim=[-3, -2, -1])
            
            l1 = min(self.alpha, Nx//2+1)
            l2 = min(self.alpha, Ny//2+1)
            out_ft = torch.zeros(B, c*k_sq, Nx, Ny, T//2 + 1, device=x.device, dtype=torch.cfloat)
            
            out_ft[:, :, :l1, :l2, :self.alpha] = torch.einsum("bixyz,ioxyz->boxyz", 
                x_fft[:, :, :l1, :l2, :self.alpha], self.weights1[:, :, :l1, :l2, :])
            out_ft[:, :, -l1:, :l2, :self.alpha] = torch.einsum("bixyz,ioxyz->boxyz", 
                x_fft[:, :, -l1:, :l2, :self.alpha], self.weights2[:, :, :l1, :l2, :])
            out_ft[:, :, :l1, -l2:, :self.alpha] = torch.einsum("bixyz,ioxyz->boxyz", 
                x_fft[:, :, :l1, -l2:, :self.alpha], self.weights3[:, :, :l1, :l2, :])
            out_ft[:, :, -l1:, -l2:, :self.alpha] = torch.einsum("bixyz,ioxyz->boxyz", 
                x_fft[:, :, -l1:, -l2:, :self.alpha], self.weights4[:, :, :l1, :l2, :])
            
            x = torch.fft.irfftn(out_ft, s=(Nx, Ny, T))
            x = x.permute(0, 2, 3, 4, 1)
            x = F.relu(x)
            x = self.output_proj(x)
            x = x.view(original_shape)
        
        return x


class MWT_CZ(nn.Module):
    """Dimension-agnostic MWT core layer"""
    
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
        """Register wavelet filters"""
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
        """Dimension-agnostic wavelet transform"""
        if self.n_dim == 1:
            xa = torch.cat([x[:, ::2, :, :], x[:, 1::2, :, :]], -1)
        elif self.n_dim >= 2:
            xa = torch.cat([x[..., ::2, ::2, :, :], x[..., ::2, 1::2, :, :], 
                           x[..., 1::2, ::2, :, :], x[..., 1::2, 1::2, :, :]], -1)
        
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s
    
    def even_odd_reconstruction(self, x):
        """Dimension-agnostic even-odd reconstruction"""
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
            
        elif self.n_dim == 3:
            B, Nx, Ny, T, c, ich = x.shape
            assert ich == 2*self.k**2
            x_ee = torch.matmul(x, self.rc_ee)
            x_eo = torch.matmul(x, self.rc_eo)
            x_oe = torch.matmul(x, self.rc_oe)
            x_oo = torch.matmul(x, self.rc_oo)
            
            result = torch.zeros(B, Nx*2, Ny*2, T, c, self.k**2, device=x.device)
            result[:, ::2, ::2, :, :, :] = x_ee
            result[:, ::2, 1::2, :, :, :] = x_eo
            result[:, 1::2, ::2, :, :, :] = x_oe
            result[:, 1::2, 1::2, :, :, :] = x_oo
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
            elif self.n_dim == 3:
                T = spatial_dims[2]
                x = self.T0(x.view(B, coarse_size, coarse_size, T, -1)).view(B, coarse_size, coarse_size, T, c, ich)
        
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.even_odd_reconstruction(x)
        
        return x


class MWT(nn.Module):
    """
    Parameters
    ----------
    n_modes : Tuple[int] or int
        If int, creates square modes for corresponding dimension (1D: alpha, 2D: (alpha, alpha), 3D: (alpha, alpha, alpha))
        If Tuple, uses specified modes directly
    in_channels : int
        Number of input function channels
    out_channels : int
        Number of output function channels
    k : int, default=3
        Wavelet kernel size
    c : int, default=1
        Number of channels in wavelet transform
    n_layers : int, default=3
        Number of MWT_CZ layers
    L : int, default=0
        Number of coarsest scale layers to skip
    lifting_channels : int, default=128
        Lifting layer channels, if 0 uses simple linear transformation
    projection_channels : int, default=128
        Projection layer channels, if 0 uses simple linear transformation
    base : str, default='legendre'
        Wavelet basis type ('legendre', 'chebyshev')
    initializer : callable, optional
        Weight initialization function
    
    Examples
    --------
    >>> model_1d = MWT(alpha=5, in_channels=1, out_channels=1)
    >>> model_2d = MWT((5, 5), in_channels=1, out_channels=1)
    >>> model_3d = MWT(alpha=5, n_dim=3, in_channels=1, out_channels=1)
    """
    
    def __init__(
        self,
        n_modes=None,
        alpha=None,
        n_dim=None,
        in_channels: int = 1,
        out_channels: int = 1,
        k: int = 3,
        c: int = 1,
        n_layers: int = 3,
        L: int = 0,
        lifting_channels: int = 128,
        projection_channels: int = 128,
        base: str = 'legendre',
        initializer=None,
        **kwargs
    ):
        super().__init__()
        
        if n_modes is not None:
            if isinstance(n_modes, (tuple, list)):
                self.n_modes = tuple(n_modes)
                self.n_dim = len(self.n_modes)
                alpha = n_modes[0]
            else:
                alpha = n_modes
                self.n_dim = 1
                self.n_modes = (alpha,)
        elif alpha is not None:
            if n_dim is None:
                self.n_dim = 1
            else:
                self.n_dim = n_dim
            self.n_modes = tuple([alpha] * self.n_dim)
        else:
            raise ValueError("Either n_modes or alpha must be specified")
        
        if self.n_dim not in [1, 2, 3]:
            raise ValueError(f"MWT only supports 1D, 2D, and 3D. Got {self.n_dim}D")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.c = c
        self.n_layers = n_layers
        self.L = L
        self.base = base
        self.alpha = alpha
        
        if self.n_dim == 1:
            ich_multiplier = c * k
        else:
            ich_multiplier = c * k**2
        
        self.lifting = self._build_lifting_layer(in_channels, ich_multiplier, lifting_channels)
        
        self.mwt_layers = nn.ModuleList([
            MWT_CZ(k=k, alpha=alpha, L=L, c=c, base=base, n_dim=self.n_dim, initializer=initializer)
            for _ in range(n_layers)
        ])
        
        self.projection = self._build_projection_layer(ich_multiplier, out_channels, projection_channels)
        
        if initializer is not None:
            self._reset_parameters(initializer)
    
    def _build_lifting_layer(self, in_channels, ich_multiplier, lifting_channels):
        """Build lifting layer"""
        if lifting_channels > 0:
            return nn.Sequential(
                nn.Linear(in_channels, lifting_channels),
                nn.ReLU(inplace=True),
                nn.Linear(lifting_channels, ich_multiplier)
            )
        else:
            return nn.Linear(in_channels, ich_multiplier)
    
    def _build_projection_layer(self, ich_multiplier, out_channels, projection_channels):
        """Build projection layer"""
        if projection_channels > 0:
            return nn.Sequential(
                nn.Linear(ich_multiplier, projection_channels),
                nn.ReLU(inplace=True),
                nn.Linear(projection_channels, out_channels)
            )
        else:
            return nn.Sequential(
                nn.Linear(ich_multiplier, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, out_channels)
            )
    
    def _reset_parameters(self, initializer):
        """Reset parameters"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                initializer(module.weight)
    
    def forward(self, x):
        """
        Forward propagation
        
        Parameters
        ----------
        x : tensor
            Input tensor with shape:
            - 1D: (B, N, in_channels)
            - 2D: (B, Nx, Ny, in_channels)  
            - 3D: (B, Nx, Ny, T, in_channels)
            
        Returns
        -------
        output : tensor
            Output tensor with shape:
            - 1D: (B, N, out_channels) or (B, N) if out_channels=1
            - 2D: (B, Nx, Ny, out_channels) or (B, Nx, Ny) if out_channels=1
            - 3D: (B, Nx, Ny, T, out_channels) or (B, Nx, Ny, T) if out_channels=1
        """
        
        x = self.lifting(x)
        
        if self.n_dim == 1:
            B, N, _ = x.shape
            x = x.view(B, N, self.c, self.k)
        elif self.n_dim == 2:
            B, Nx, Ny, _ = x.shape
            x = x.view(B, Nx, Ny, self.c, self.k**2)
        elif self.n_dim == 3:
            B, Nx, Ny, T, _ = x.shape
            x = x.view(B, Nx, Ny, T, self.c, self.k**2)
        
        for i, layer in enumerate(self.mwt_layers):
            x = layer(x)
            if i < self.n_layers - 1:
                x = F.relu(x)
        
        if self.n_dim == 1:
            x = x.view(B, N, -1)
        elif self.n_dim == 2:
            x = x.view(B, Nx, Ny, -1)
        elif self.n_dim == 3:
            x = x.view(B, Nx, Ny, T, -1)
        
        x = self.projection(x)
        
        if self.out_channels == 1:
            return x.squeeze(-1)
        return x

"""Compatible MWT Model"""

class MWT1d(MWT):
    """1D Multi-wavelet Transform Neural Operator"""
    def __init__(self, alpha: int, **kwargs):
        super().__init__(alpha=alpha, n_dim=1, **kwargs)


class MWT2d(MWT):
    """2D Multi-wavelet Transform Neural Operator"""
    def __init__(self, alpha: int, **kwargs):
        super().__init__(alpha=alpha, n_dim=2, **kwargs)


class MWT3d(MWT):
    """3D Multi-wavelet Transform Neural Operator"""
    def __init__(self, alpha: int, **kwargs):
        super().__init__(alpha=alpha, n_dim=3, **kwargs)
