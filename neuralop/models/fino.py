###
# Can not be pushed directly in the NeuralOperator 
###
from torch import nn
import torch
import itertools
import torch_harmonics as th
import tensorly as tl
from tensorly.plugins import use_opt_einsum
from .spectral_convolution import FactorizedSpectralConv

class SpectralConvKernel2d(FactorizedSpectralConv):
    '''
    fft_type = {'sph', 'norm'}, if 'sph' it uses the speherical Fourier Transform 
    '''
    def __init__(self, in_channels, out_channels, n_modes, incremental_n_modes=None, bias=True,
                 n_layers=1, separable=False, output_scaling_factor=None,
                 rank=0.5, factorization='cp', implementation='reconstructed', 
                 fixed_rank_modes=False, joint_factorization=False, decomposition_kwargs=dict(),
                 init_std='auto', fft_norm='forward', fft_type = 'sht',  sht_nlat = 180, sht_nlon = 360,\
                 sht_grid="legendre-gauss", isht_grid="legendre-gauss", sht_norm="backward", frequency_mixer = False):
        super().__init__(in_channels, out_channels, n_modes, incremental_n_modes, bias,
                 n_layers, separable, output_scaling_factor,
                 rank, factorization, implementation, 
                 fixed_rank_modes, joint_factorization, decomposition_kwargs,
                 init_std, fft_norm)

        #self.shared = shared
        
        hm1, hm2 = self.half_n_modes[0], self.half_n_modes[1]
        if self.factorization is None or not frequency_mixer:
            self.W1 = nn.Parameter(torch.empty(hm1, hm2, hm1, hm2, dtype = torch.cfloat))
            self.W2 = nn.Parameter(torch.empty(hm1, hm2, hm1, hm2, dtype = torch.cfloat))
            self.reset_parameter()
        self.sht_grid = sht_grid
        self.isht_grid = isht_grid
        self.sht_norm = sht_norm
        self.fft_type = fft_type
        self.frequency_mixer = frequency_mixer
        
        ####
        # This following line might have a version dependent nature
        ####
        if self.output_scaling_factor is not None:
            out_nlat = int(round(sht_nlat * self.output_scaling_factor[0][0]))
            out_nlon = int(round(sht_nlon * self.output_scaling_factor[0][1]))
        

        if fft_type == 'sht':
            self.forward_fft = th.RealSHT(sht_nlat, sht_nlon, grid = self.sht_grid, norm= self.sht_norm)
            self.inverse_fft = th.InverseRealSHT(out_nlat, out_nlon, grid = self.isht_grid, norm= self.sht_norm)
            

    def reset_parameter(self):
        # Initial model parameters.
        scaling_factor = 1/(self.in_channels * self.half_n_modes[0]*self.half_n_modes[1])**0.5
        torch.nn.init.normal_(self.W1, mean=0.0, std=scaling_factor)
        torch.nn.init.normal_(self.W2, mean=0.0, std=scaling_factor)
    
    def mode_mixer(self, input, weights):
        return torch.einsum("bimn,mnop->biop", input, weights)
    
    def forward(self, x, indices=0, output_shape = None):
        batchsize, channels, height, width = x.shape

        if self.fft_type == 'sht':
            if self.forward_fft.nlat != x.shape[-2] or self.forward_fft.nlon != x.shape[-1]:
                self.forward_fft = th.RealSHT(x.shape[-2], x.shape[-1], grid = self.sht_grid, norm= self.sht_norm).to(x.device)
            x = self.forward_fft(x.double()).to(dtype = torch.cfloat)
        else:
            x = torch.fft.rfft2(x.float(), norm=self.fft_norm)

        #mode mixer
        #uses separate MLP to mix mode along each co-dim/channels
    

        if self.factorization is None or not self.frequency_mixer:
            x[:,:, :self.half_n_modes[0], :self.half_n_modes[1]] = self.mode_mixer(x[:,:, :self.half_n_modes[0], :self.half_n_modes[1]].clone(), self.W1)
            x[:,:, -self.half_n_modes[0]:, :self.half_n_modes[1]] = self.mode_mixer(x[:,:, -self.half_n_modes[0]:, :self.half_n_modes[1]].clone(), self.W2)
        

        #spectral conv / channel mixer
        
        # The output will be of size (batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1)
        out_fft = torch.zeros([batchsize, self.out_channels, height, width//2 + 1], dtype=x.dtype, device=x.device)

        # upper block (truncate high freq)
        out_fft[:, :, :self.half_n_modes[0], :self.half_n_modes[1]] = self._contract(x[:, :, :self.half_n_modes[0], :self.half_n_modes[1]], 
                                                                              self._get_weight(2*indices), separable=self.separable)
        # Lower block
        out_fft[:, :, -self.half_n_modes[0]:, :self.half_n_modes[1]] = self._contract(x[:, :, -self.half_n_modes[0]:, :self.half_n_modes[1]],
                                                                              self._get_weight(2*indices + 1), separable=self.separable)
            
        if self.output_scaling_factor is not None and output_shape is None:
            height = int(round(height*self.output_scaling_factor[indices][0]))
            width = int(round(width*self.output_scaling_factor[indices][1]))
            
        if output_shape is not None:
            height = output_shape[0]
            width = output_shape[1]

        if self.fft_type == 'sht':
            if self.inverse_fft.lmax != out_fft.shape[-2] or self.inverse_fft.mmax != out_fft.shape[-1] or self.inverse_fft.nlat != height or self.inverse_fft.nlon != width :
                 self.inverse_fft = th.InverseRealSHT(height, width, lmax= out_fft.shape[-2], mmax= out_fft.shape[-1], grid = self.sht_grid, norm= self.sht_norm).to(x.device)
            x = self.inverse_fft(out_fft.to(dtype= torch.cdouble)).float()
        else:
            x = torch.fft.irfft2(out_fft, s=(height, width), dim=(-2, -1), norm=self.fft_norm)        

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x
