import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


from typing import List, Tuple, Union, Literal
import math

from ..mwt_utils import get_filter

class MWT(nn.Module):
    """N-Dimensional Multiwavelet Transform Neural Operator. The MWT learns a mapping between
    spaces of functions discretized over regular grids using multiscale wavelet transforms
    combined with sparse kernels and Fourier transforms.
    
    The key component of an MWT is its multiscale wavelet decomposition and reconstruction
    process, which efficiently captures both local and global features at different scales.

    Parameters
    ----------
    n_modes : Tuple[int]
        determines the structure based on the number of dimensions:
        - If len(n_modes) == 1: creates 1D MWT
        - If len(n_modes) == 2: creates 2D MWT  
        - If len(n_modes) == 3: creates 3D MWT
        Each value is used as the alpha parameter for sparse kernels
    in_channels : int
        Number of channels in input function
    out_channels : int
        Number of channels in output function
    k : int, optional
        Size of the wavelet kernel, by default 3
    c : int, optional
        Number of channels in the wavelet transform, by default 1
    n_layers : int, optional
        Number of MWT_CZ layers, by default 3
    L : int, optional
        Number of decomposition levels to skip (coarsest scale), by default 0

    Other parameters
    ----------------
    lifting_channels : int, optional
        Number of channels in the lifting layer, by default 128
        If 0, uses a simple linear transformation
    projection_channels : int, optional
        Number of channels in the projection layer, by default 128
        If 0, uses a simple linear transformation
    base : str, optional
        Type of wavelet basis to use, by default 'legendre'
        Options include 'legendre', 'db', 'sym', etc.
    initializer : callable, optional
        Weight initialization function, by default None
        If provided, will be used to initialize the linear layer weights

    Examples
    --------
    
    >>> from models import MWT
    >>> # 1D MWT
    >>> model_1d = MWT(n_modes=(5,), in_channels=1, out_channels=1, k=3)
    >>> 
    >>> # 2D MWT
    >>> model_2d = MWT(n_modes=(5, 5), in_channels=1, out_channels=1, k=3)
    >>> 
    >>> # 3D MWT  
    >>> model_3d = MWT(n_modes=(5, 5, 5), in_channels=1, out_channels=1, k=3)

    References
    ----------
    .. [1] :
    
    Gupta, G. et al. "Multiwavelet-based Operator Learning for Differential Equations" 
        NeurIPS 2021, https://arxiv.org/abs/2109.13459
    """

    def __init__(
        self,
        n_modes: Tuple[int],
        in_channels: int,
        out_channels: int,
        k: int = 3,
        c: int = 1,
        n_layers: int = 3,
        L: int = 0,
        lifting_channels: int = 128,
        projection_channels: int = 128,
        base: str = 'legendre',
        initializer = None,
        **kwargs
    ):
        super(MWT, self).__init__()
        
        self.n_dim = len(n_modes)
        self.n_modes = n_modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.c = c
        self.n_layers = n_layers
        self.L = L
        self.base = base
        
        # Determine alpha based on dimensionality
        if self.n_dim == 1:
            alpha = n_modes[0]
            ich_multiplier = c * k
            MWT_CZ_class = MWT_CZ1d
        elif self.n_dim == 2:
            alpha = n_modes[0]  # Assuming square modes
            ich_multiplier = c * k**2
            MWT_CZ_class = MWT_CZ2d
        elif self.n_dim == 3:
            alpha = n_modes[0]  # Assuming cubic modes
            ich_multiplier = c * k**2
            MWT_CZ_class = MWT_CZ3d
        else:
            raise ValueError(f"MWT only supports 1D, 2D, and 3D. Got {self.n_dim}D")
        
        # Lifting layer
        if lifting_channels > 0:
            self.Lk = nn.Sequential(
                nn.Linear(in_channels, lifting_channels),
                nn.ReLU(inplace=True),
                nn.Linear(lifting_channels, ich_multiplier)
            )
        else:
            self.Lk = nn.Linear(in_channels, ich_multiplier)
        
        # MWT_CZ layers
        self.MWT_CZ = nn.ModuleList(
            [MWT_CZ_class(k, alpha, L, c, base, initializer) 
             for _ in range(n_layers)]
        )
        
        # Projection layers
        if projection_channels > 0:
            self.Lc0 = nn.Linear(ich_multiplier, projection_channels)
            self.Lc1 = nn.Linear(projection_channels, out_channels)
        else:
            self.Lc0 = nn.Linear(ich_multiplier, 128)
            self.Lc1 = nn.Linear(128, out_channels)
        
        if initializer is not None:
            self.reset_parameters(initializer)
    
    def forward(self, x):
        """MWT's forward pass
        
        1. Lifts input to high-dimensional space suitable for wavelet transform
        
        2. Applies `n_layers` MWT_CZ layers in sequence, each performing:
           - Multi-level wavelet decomposition
           - Sparse kernel operations in both spatial and Fourier domains
           - Multi-level wavelet reconstruction
           
        3. Projects the result back to the output dimension

        Parameters
        ----------
        x : tensor
            Input tensor with shape:
            - 1D: (B, N, in_channels)
            - 2D: (B, Nx, Ny, in_channels)
            - 3D: (B, Nx, Ny, Nt, in_channels)
            
        Returns
        -------
        output : tensor
            Output tensor with shape:
            - 1D: (B, N) if out_channels=1, else (B, N, out_channels)
            - 2D: (B, Nx, Ny) if out_channels=1, else (B, Nx, Ny, out_channels)
            - 3D: (B, Nx, Ny, Nt) if out_channels=1, else (B, Nx, Ny, Nt, out_channels)
        """
        
        # Get input shape based on dimensionality
        if self.n_dim == 1:
            B, N, ich = x.shape
            x = self.Lk(x)
            x = x.view(B, N, self.c, self.k)
        elif self.n_dim == 2:
            B, Nx, Ny, ich = x.shape
            x = self.Lk(x)
            x = x.view(B, Nx, Ny, self.c, self.k**2)
        elif self.n_dim == 3:
            B, Nx, Ny, T, ich = x.shape
            x = self.Lk(x)
            x = x.view(B, Nx, Ny, T, self.c, self.k**2)
    
        # Apply MWT_CZ layers
        for i in range(self.n_layers):
            x = self.MWT_CZ[i](x)
            if i < self.n_layers - 1:
                x = F.relu(x)

        # Reshape and project to output
        if self.n_dim == 1:
            x = x.view(B, N, -1)
        elif self.n_dim == 2:
            x = x.view(B, Nx, Ny, -1)
        elif self.n_dim == 3:
            x = x.view(B, Nx, Ny, T, -1)

        #TODO: Check if this can be added?            
        x = self.Lc0(x)
        x = F.relu(x)
        x = self.Lc1(x)
        
        # Squeeze if single output channel
        if self.out_channels == 1:
            return x.squeeze(-1)
        return x
    
    def reset_parameters(self, initializer):
        """Reset parameters using the provided initializer"""
        initializer(self.Lc0.weight)
        initializer(self.Lc1.weight)
        if hasattr(self.Lk, 'weight'):
            initializer(self.Lk.weight)
        elif hasattr(self.Lk, 'modules'):  # Sequential
            for module in self.Lk.modules():
                if isinstance(module, nn.Linear):
                    initializer(module.weight)


class sparseKernel1d(nn.Module):
    def __init__(self,
                 k, alpha, c=1,
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernel1d,self).__init__()
       
        self.k = k
        self.Li = nn.Linear(c*k, 128)
        self.conv = self.convBlock(c*k, 128)
        self.Lo = nn.Linear(128, c*k)
       
    def forward(self, x):
        B, N, c, ich = x.shape # (B, N, c, k)
        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.Lo(x)
        x = x.view(B, N, c, ich)
        return x
       
       
    def convBlock(self, ich, och):
        net = nn.Sequential(
            nn.Conv1d(ich, och, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        return net

def compl_mul1d(x, weights):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", x, weights)

class sparseKernelFT1d(nn.Module):
    def __init__(self,
                 k, alpha, c=1,
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernelFT1d, self).__init__()       
       
        self.modes1 = alpha
        self.scale = (1 / (c*k*c*k))
        self.weights1 = nn.Parameter(self.scale * torch.rand(c*k, c*k, self.modes1, dtype=torch.cfloat))
        self.weights1.requires_grad = True
        self.k = k
       
    def forward(self, x):
        B, N, c, k = x.shape # (B, N, c, k)
       
        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        l = min(self.modes1, N//2+1)
        out_ft = torch.zeros(B, c*k, N//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = compl_mul1d(x_fft[:, :, :l], self.weights1[:, :, :l])

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=N)
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return x


class MWT_CZ1d(nn.Module):
    def __init__(self,
                 k = 3, alpha = 5,
                 L = 0, c = 1,
                 base = 'legendre',
                 initializer = None,
                 **kwargs):
        super(MWT_CZ1d, self).__init__()
       
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0@PHI0
        G0r = G0@PHI0
        H1r = H1@PHI1
        G1r = G1@PHI1
        
        H0r[np.abs(H0r)<1e-8]=0
        H1r[np.abs(H1r)<1e-8]=0
        G0r[np.abs(G0r)<1e-8]=0
        G1r[np.abs(G1r)<1e-8]=0
       
        self.A = sparseKernelFT1d(k, alpha, c)
        self.B = sparseKernelFT1d(k, alpha, c)
        self.C = sparseKernelFT1d(k, alpha, c)
       
        self.T0 = nn.Linear(k, k)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((G0.T, G1.T), axis=0)))
       
        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((H1r, G1r), axis=0)))
       
       
    def forward(self, x):
       
        B, N, c, ich = x.shape # (B, N, k)
        ns = math.floor(np.log2(N))

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])
#         decompose
        for i in range(ns-self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x) # coarsest scale transform

#        reconstruct           
        for i in range(ns-1-self.L,-1,-1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)
        return x

   
    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                       ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s
       
       
    def evenOdd(self, x):
       
        B, N, c, ich = x.shape # (B, N, c, k)
        assert ich == 2*self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)
       
        x = torch.zeros(B, N*2, c, self.k,
            device = x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x
   
   
class sparseKernel2d(nn.Module):
    def __init__(self,
                 k, alpha, c=1, 
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernel2d,self).__init__()
        
        self.k = k
        self.conv = self.convBlock(k, c*k**2, alpha)
        self.Lo = nn.Linear(alpha*k**2, c*k**2)
        
    def forward(self, x):
        B, Nx, Ny, c, ich = x.shape # (B, Nx, Ny, c, k**2)
        x = x.view(B, Nx, Ny, -1)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.Lo(x)
        x = x.view(B, Nx, Ny, c, ich)
        
        return x
        
        
    def convBlock(self, k, W, alpha):
        och = alpha * k**2
        net = nn.Sequential(
            nn.Conv2d(W, och, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        return net 
    
    
def compl_mul2d(x, weights):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    return torch.einsum("bixy,ioxy->boxy", x, weights)


class sparseKernelFT2d(nn.Module):
    def __init__(self,
                 k, alpha, c=1, 
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernelFT2d, self).__init__()        
        
        self.modes = alpha

        self.weights1 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, dtype=torch.cfloat))        
        nn.init.xavier_normal_(self.weights1)
        nn.init.xavier_normal_(self.weights2)
        
        self.Lo = nn.Linear(c*k**2, c*k**2)
        self.k = k
        
    def forward(self, x):
        B, Nx, Ny, c, ich = x.shape # (B, N, N, c, k^2)
        
        x = x.view(B, Nx, Ny, -1)
        x = x.permute(0, 3, 1, 2)
        x_fft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        l1 = min(self.modes, Nx//2+1)
        l1l = min(self.modes, Nx//2-1)
        l2 = min(self.modes, Ny//2+1)
        out_ft = torch.zeros(B, c*ich, Nx, Ny//2 + 1,  device=x.device, dtype=torch.cfloat)
        
        out_ft[:, :, :l1, :l2] = compl_mul2d(
            x_fft[:, :, :l1, :l2], self.weights1[:, :, :l1, :l2])
        out_ft[:, :, -l1:, :l2] = compl_mul2d(
                x_fft[:, :, -l1:, :l2], self.weights2[:, :, :l1, :l2])
        
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s = (Nx, Ny))
        
        x = x.permute(0, 2, 3, 1)
        x = F.relu(x)
        x = self.Lo(x)
        x = x.view(B, Nx, Ny, c, ich)
        return x
        
    
class MWT_CZ2d(nn.Module):
    def __init__(self,
                 k = 3, alpha = 5, 
                 L = 0, c = 1,
                 base = 'legendre',
                 initializer = None,
                 **kwargs):
        super(MWT_CZ2d, self).__init__()
        
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0@PHI0
        G0r = G0@PHI0
        H1r = H1@PHI1
        G1r = G1@PHI1
        H0r[np.abs(H0r)<1e-8]=0
        H1r[np.abs(H1r)<1e-8]=0
        G0r[np.abs(G0r)<1e-8]=0
        G1r[np.abs(G1r)<1e-8]=0

        self.A = sparseKernelFT2d(k, alpha, c)
        self.B = sparseKernel2d(k, c, c)
        self.C = sparseKernel2d(k, c, c)
        
        self.T0 = nn.Linear(c*k**2, c*k**2)

        if initializer is not None:
            self.reset_parameters(initializer)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((np.kron(H0, H0).T, 
                            np.kron(H0, H1).T,
                            np.kron(H1, H0).T,
                            np.kron(H1, H1).T,
                           ), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((np.kron(G0, G0).T,
                            np.kron(G0, G1).T,
                            np.kron(G1, G0).T,
                            np.kron(G1, G1).T,
                           ), axis=0)))
        
        self.register_buffer('rc_ee', torch.Tensor(
            np.concatenate((np.kron(H0r, H0r), 
                            np.kron(G0r, G0r),
                           ), axis=0)))
        self.register_buffer('rc_eo', torch.Tensor(
            np.concatenate((np.kron(H0r, H1r), 
                            np.kron(G0r, G1r),
                           ), axis=0)))
        self.register_buffer('rc_oe', torch.Tensor(
            np.concatenate((np.kron(H1r, H0r), 
                            np.kron(G1r, G0r),
                           ), axis=0)))
        self.register_buffer('rc_oo', torch.Tensor(
            np.concatenate((np.kron(H1r, H1r), 
                            np.kron(G1r, G1r),
                           ), axis=0)))
        
        
    def forward(self, x):
        
        B, Nx, Ny, c, ich = x.shape # (B, Nx, Ny, c, k**2)
        ns = math.floor(np.log2(Nx))

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

#         decompose
        for i in range(ns-self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x.view(B, 2**self.L, 2**self.L, -1)).view(
            B, 2**self.L, 2**self.L, c, ich) # coarsest scale transform

#        reconstruct            
        for i in range(ns-1-self.L,-1,-1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)

        return x

    
    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2 , ::2 , :, :], 
                        x[:, ::2 , 1::2, :, :], 
                        x[:, 1::2, ::2 , :, :], 
                        x[:, 1::2, 1::2, :, :]
                       ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s
        
        
    def evenOdd(self, x):
        
        B, Nx, Ny, c, ich = x.shape # (B, Nx, Ny, c, k**2)
        assert ich == 2*self.k**2
        x_ee = torch.matmul(x, self.rc_ee)
        x_eo = torch.matmul(x, self.rc_eo)
        x_oe = torch.matmul(x, self.rc_oe)
        x_oo = torch.matmul(x, self.rc_oo)
        
        x = torch.zeros(B, Nx*2, Ny*2, c, self.k**2, 
            device = x.device)
        x[:, ::2 , ::2 , :, :] = x_ee
        x[:, ::2 , 1::2, :, :] = x_eo
        x[:, 1::2, ::2 , :, :] = x_oe
        x[:, 1::2, 1::2, :, :] = x_oo
        return x
    
    def reset_parameters(self, initializer):
        initializer(self.T0.weight)
    
    
class sparseKernel3d(nn.Module):
    def __init__(self,
                 k, alpha, c=1, 
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernel3d,self).__init__()
        
        self.k = k
        self.conv = self.convBlock(alpha*k**2, alpha*k**2)
        self.Lo = nn.Linear(alpha*k**2, c*k**2)
        
    def forward(self, x):
        B, Nx, Ny, T, c, ich = x.shape # (B, Nx, Ny, T, c, k**2)
        x = x.view(B, Nx, Ny, T, -1)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.Lo(x)
        x = x.view(B, Nx, Ny, T, c, ich)
        
        return x
        
        
    def convBlock(self, ich, och):
        net = nn.Sequential(
            nn.Conv3d(och, och, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        return net 
    
    
def compl_mul3d(input, weights):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixyz,ioxyz->boxyz", input, weights)


class sparseKernelFT3d(nn.Module):
    def __init__(self,
                 k, alpha, c=1, 
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernelFT3d, self).__init__()        
        
        self.modes = alpha

        self.weights1 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, self.modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, self.modes, dtype=torch.cfloat))        
        self.weights3 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, self.modes, dtype=torch.cfloat))        
        self.weights4 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, self.modes, dtype=torch.cfloat))        
        nn.init.xavier_normal_(self.weights1)
        nn.init.xavier_normal_(self.weights2)
        nn.init.xavier_normal_(self.weights3)
        nn.init.xavier_normal_(self.weights4)
        
        self.Lo = nn.Linear(c*k**2, c*k**2)
        self.k = k
        
    def forward(self, x):
        B, Nx, Ny, T, c, ich = x.shape # (B, N, N, T, c, k^2)
        
        x = x.view(B, Nx, Ny, T, -1)
        x = x.permute(0, 4, 1, 2, 3)
        x_fft = torch.fft.rfftn(x, dim = [-3, -2, -1])
        
        # Multiply relevant Fourier modes
        l1 = min(self.modes, Nx//2+1)
        l2 = min(self.modes, Ny//2+1)
        out_ft = torch.zeros(B, c*ich, Nx, Ny, T//2 +1, device=x.device, dtype=torch.cfloat)
        
        out_ft[:, :, :l1, :l2, :self.modes] = compl_mul3d(
            x_fft[:, :, :l1, :l2, :self.modes], self.weights1[:, :, :l1, :l2, :])
        out_ft[:, :, -l1:, :l2, :self.modes] = compl_mul3d(
                x_fft[:, :, -l1:, :l2, :self.modes], self.weights2[:, :, :l1, :l2, :])
        out_ft[:, :, :l1, -l2:, :self.modes] = compl_mul3d(
                x_fft[:, :, :l1, -l2:, :self.modes], self.weights3[:, :, :l1, :l2, :])
        out_ft[:, :, -l1:, -l2:, :self.modes] = compl_mul3d(
                x_fft[:, :, -l1:, -l2:, :self.modes], self.weights4[:, :, :l1, :l2, :])
        
        #Return to physical space
        x = torch.fft.irfftn(out_ft, s = (Nx, Ny, T))
        
        x = x.permute(0, 2, 3, 4, 1)
        x = F.relu(x)
        x = self.Lo(x)
        x = x.view(B, Nx, Ny, T, c, ich)
        return x
        
    
class MWT_CZ3d(nn.Module):
    def __init__(self,
                 k = 3, alpha = 5, 
                 L = 0, c = 1,
                 base = 'legendre',
                 initializer = None,
                 **kwargs):
        super(MWT_CZ3d, self).__init__()
        
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0@PHI0
        G0r = G0@PHI0
        H1r = H1@PHI1
        G1r = G1@PHI1
        
        H0r[np.abs(H0r)<1e-8]=0
        H1r[np.abs(H1r)<1e-8]=0
        G0r[np.abs(G0r)<1e-8]=0
        G1r[np.abs(G1r)<1e-8]=0

        self.A = sparseKernelFT3d(k, alpha, c)
        self.B = sparseKernel3d(k, c, c)
        self.C = sparseKernel3d(k, c, c)
        
        self.T0 = nn.Linear(c*k**2, c*k**2)

        if initializer is not None:
            self.reset_parameters(initializer)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((np.kron(H0, H0).T, 
                            np.kron(H0, H1).T,
                            np.kron(H1, H0).T,
                            np.kron(H1, H1).T,
                           ), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((np.kron(G0, G0).T,
                            np.kron(G0, G1).T,
                            np.kron(G1, G0).T,
                            np.kron(G1, G1).T,
                           ), axis=0)))
        
        self.register_buffer('rc_ee', torch.Tensor(
            np.concatenate((np.kron(H0r, H0r), 
                            np.kron(G0r, G0r),
                           ), axis=0)))
        self.register_buffer('rc_eo', torch.Tensor(
            np.concatenate((np.kron(H0r, H1r), 
                            np.kron(G0r, G1r),
                           ), axis=0)))
        self.register_buffer('rc_oe', torch.Tensor(
            np.concatenate((np.kron(H1r, H0r), 
                            np.kron(G1r, G0r),
                           ), axis=0)))
        self.register_buffer('rc_oo', torch.Tensor(
            np.concatenate((np.kron(H1r, H1r), 
                            np.kron(G1r, G1r),
                           ), axis=0)))
        
        
    def forward(self, x):
        
        B, Nx, Ny, T, c, ich = x.shape # (B, Nx, Ny, T, c, k**2)
        ns = math.floor(np.log2(Nx))

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

#         decompose
        for i in range(ns-self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x.view(B, 2**self.L, 2**self.L, T, -1)).view(
            B, 2**self.L, 2**self.L, T, c, ich) # coarsest scale transform

#        reconstruct            
        for i in range(ns-1-self.L,-1,-1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)

        return x

    
    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2 , ::2 , :, :, :], 
                        x[:, ::2 , 1::2, :, :, :], 
                        x[:, 1::2, ::2 , :, :, :], 
                        x[:, 1::2, 1::2, :, :, :]
                       ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s
        
        
    def evenOdd(self, x):
        
        B, Nx, Ny, T, c, ich = x.shape # (B, Nx, Ny, c, k**2)
        assert ich == 2*self.k**2
        x_ee = torch.matmul(x, self.rc_ee)
        x_eo = torch.matmul(x, self.rc_eo)
        x_oe = torch.matmul(x, self.rc_oe)
        x_oo = torch.matmul(x, self.rc_oo)
        
        x = torch.zeros(B, Nx*2, Ny*2, T, c, self.k**2, 
            device = x.device)
        x[:, ::2 , ::2 , :, :, :] = x_ee
        x[:, ::2 , 1::2, :, :, :] = x_eo
        x[:, 1::2, ::2 , :, :, :] = x_oe
        x[:, 1::2, 1::2, :, :, :] = x_oo
        return x
    
    def reset_parameters(self, initializer):
        initializer(self.T0.weight)




class MWT1d(MWT):
    """1D Multiwavelet Transform Neural Operator

    For the full list of parameters, see :class:`MWT`.

    Parameters
    ----------
    alpha : int
        Number of Fourier modes to use in sparse kernels
    """

    def __init__(
        self,
        alpha: int,
        in_channels: int = 1,
        out_channels: int = 1,
        k: int = 3,
        c: int = 1,
        n_layers: int = 3,
        L: int = 0,
        lifting_channels: int = 128,
        projection_channels: int = 128,
        base: str = 'legendre',
        initializer = None,
        **kwargs
    ):
        super().__init__(
            n_modes=(alpha,),
            in_channels=in_channels,
            out_channels=out_channels,
            k=k,
            c=c,
            n_layers=n_layers,
            L=L,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            base=base,
            initializer=initializer,
            **kwargs
        )
        self.alpha = alpha


class MWT2d(MWT):
    """2D Multiwavelet Transform Neural Operator

    For the full list of parameters, see :class:`MWT`.

    Parameters
    ----------
    alpha : int
        Number of Fourier modes to use in sparse kernels
    """

    def __init__(
        self,
        alpha: int,
        in_channels: int = 1,
        out_channels: int = 1,
        k: int = 3,
        c: int = 1,
        n_layers: int = 3,
        L: int = 0,
        lifting_channels: int = 128,
        projection_channels: int = 128,
        base: str = 'legendre',
        initializer = None,
        **kwargs
    ):
        super().__init__(
            n_modes=(alpha, alpha),
            in_channels=in_channels,
            out_channels=out_channels,
            k=k,
            c=c,
            n_layers=n_layers,
            L=L,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            base=base,
            initializer=initializer,
            **kwargs
        )
        self.alpha = alpha


class MWT3d(MWT):
    """3D Multiwavelet Transform Neural Operator

    For the full list of parameters, see :class:`MWT`.

    Parameters
    ----------
    alpha : int
        Number of Fourier modes to use in sparse kernels
    """

    def __init__(
        self,
        alpha: int,
        in_channels: int = 1,
        out_channels: int = 1,
        k: int = 3,
        c: int = 1,
        n_layers: int = 3,
        L: int = 0,
        lifting_channels: int = 128,
        projection_channels: int = 128,
        base: str = 'legendre',
        initializer = None,
        **kwargs
    ):
        super().__init__(
            n_modes=(alpha, alpha, alpha),
            in_channels=in_channels,
            out_channels=out_channels,
            k=k,
            c=c,
            n_layers=n_layers,
            L=L,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            base=base,
            initializer=initializer,
            **kwargs
        )
        self.alpha = alpha

