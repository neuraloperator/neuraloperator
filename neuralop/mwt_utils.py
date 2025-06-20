import torch
import torch.nn as nn

import numpy as np
from functools import partial

from scipy.special import eval_legendre
from sympy import Poly, legendre, Symbol, chebyshevt

def legendreDer(k, x):
    def _legendre(k, x):
        return (2*k+1) * eval_legendre(k, x)
    out = 0
    for i in np.arange(k-1,-1,-2):
        out += _legendre(i, x)
    return out

def phi_(phi_c, x, lb = 0, ub = 1):
    mask = np.logical_or(x<lb, x>ub) * 1.0
    return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1-mask)

def get_phi_psi(k, base):
    
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
                
        phi = [partial(phi_, phi_coeff[i,:]) for i in range(k)]
        
        x = Symbol('x')
        kUse = 2*k
        roots = Poly(chebyshevt(kUse, 2*x-1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / kUse / 2
        
        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))

        psi1 = [[] for _ in range(k)]
        psi2 = [[] for _ in range(k)]

        for ki in range(k):
            psi1_coeff[ki,:] = phi_2x_coeff[ki,:]
            for i in range(k):
                proj_ = (wm * phi[i](x_m) * np.sqrt(2)* phi[ki](2*x_m)).sum()
                psi1_coeff[ki,:] -= proj_ * phi_coeff[i,:]
                psi2_coeff[ki,:] -= proj_ * phi_coeff[i,:]

            for j in range(ki):
                proj_ = (wm * psi1[j](x_m) * np.sqrt(2) * phi[ki](2*x_m)).sum()        
                psi1_coeff[ki,:] -= proj_ * psi1_coeff[j,:]
                psi2_coeff[ki,:] -= proj_ * psi2_coeff[j,:]

            psi1[ki] = partial(phi_, psi1_coeff[ki,:], lb = 0, ub = 0.5)
            psi2[ki] = partial(phi_, psi2_coeff[ki,:], lb = 0.5, ub = 1)

            norm1 = (wm * psi1[ki](x_m) * psi1[ki](x_m)).sum()
            norm2 = (wm * psi2[ki](x_m) * psi2[ki](x_m)).sum()

            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki,:] /= norm_
            psi2_coeff[ki,:] /= norm_
            psi1_coeff[np.abs(psi1_coeff)<1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff)<1e-8] = 0

            psi1[ki] = partial(phi_, psi1_coeff[ki,:], lb = 0, ub = 0.5+1e-16)
            psi2[ki] = partial(phi_, psi2_coeff[ki,:], lb = 0.5+1e-16, ub = 1)
        
    return phi, psi1, psi2


def get_filter(base, k):
    
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
    phi, psi1, psi2 = get_phi_psi(k, base)
    if base == 'legendre':
        roots = Poly(legendre(k, 2*x-1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = 1/k/legendreDer(k,2*x_m-1)/eval_legendre(k-1,2*x_m-1)
        
        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki](x_m/2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m/2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki]((x_m+1)/2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m+1)/2) * phi[kpi](x_m)).sum()
                
        PHI0 = np.eye(k)
        PHI1 = np.eye(k)
                
    elif base == 'chebyshev':
        x = Symbol('x')
        kUse = 2*k
        roots = Poly(chebyshevt(kUse, 2*x-1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
        # not needed for our purpose here, we use even k always to avoid
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

    H0[np.abs(H0)<1e-8] = 0
    H1[np.abs(H1)<1e-8] = 0
    G0[np.abs(G0)<1e-8] = 0
    G1[np.abs(G1)<1e-8] = 0
        
    return H0, H1, G0, G1, PHI0, PHI1
