from typing import Iterable, Literal

import torch
from torch.fft import irfftn
from torch.testing import assert_close

class irfftn_handle(object):
    """ 
    Wrapper around torch.fft.irfftn() to manually correct cufft. 
    """
    def __init__(self, device):
        # set up cuda to use CUFFT backend if possible

        x = torch.randn(32,dtype=torch.cfloat).to(device)
        x_no_imag = torch.clone(x)

        # manually set zero and nyquist imaginary components to zero 
        # this should happen in the inverse RFFT but some CUFFT backends
        # do not verify this. 
        x_no_imag[0].imag = 0. 
        x_no_imag[16].imag = 0. 

        x_irfft = irfftn(x)
        x_irfft_no_imag = irfftn(x_no_imag)

        try:
            assert_close(x_irfft, x_irfft_no_imag)
            self.needs_manual_correction = False
        except AssertionError:
            self.needs_manual_correction = True
    
    def __call__(self, x: torch.Tensor, 
                 fft_dims: Iterable=None, 
                 mode_sizes: Iterable=None, 
                 norm: Literal['ortho', 'forward', 'backward']=None):
        """
        Compute the inverse Real N-dimensional Fast Fourier Transform
        
        Parameters
        ----------
        x : ``torch.Tensor``
            input tensor of basis coefficients to be inverse transformed
        fft_dims : ``Iterable``, optional
            dimensions of x along which to perform the IRFFT,
            by default None. 
            If None, performs IRFFT along each dimension. 
        mode_sizes : ``Iterable``, optional
            Size of the output of IRFFT along each dimension,
            corresponding to the dimensions listed in fft_dims,
            by default None.
            If None, uses default output size computed by IRFFT.
        norm : ``Literal['ortho', 'forward', 'backward']``, optional
            Whether to normalize the outputs of the IRFFT, by default None
            * If ``'ortho'``, scales outputs by ``1/sqrt(n)`` 
            along each dim of size ``n`` 
            * If ``'forward'``, does nothing in the reverse direction,
            as input is assumed to be scaled by ``1/n`` by the FFT.
            * If ``'backward'``, scales outputs by ``1/n`` 
            along each dim of size ``n``. 
        """
        
        if self.needs_manual_correction:
            zero_frequency_inds = [] 
            nyquist_frequency_inds = []
            fft_size = [x.shape[k] for k in fft_dims]
            
            # correct zero and optionally nyquist frequency imaginary components
            # along each transformed mode of the FFT tensor
            for mode_size, dim in zip(fft_size, fft_dims):
                zero_ind = [slice(None)] * x.ndim
                zero_ind[dim] = slice(0,1)

                x[zero_ind].imag = 0.

                # only correct the nyquist frequency here if the signal length along 
                # this dimension is a multiple of two. 
                if mode_size % 2 == 0:
                    nyquist_ind = [slice(None)] * x.ndim
                    nyquist_ind[dim] = slice(mode_size // 2, mode_size // 2 + 1)
                    x[nyquist_ind].imag = 0.
                
            # Manually set the imaginary component of the nyquist frequency 
            # of the last dim, which we truncate to remove redundant coefficients, to zero
            last_nyquist_ind = [slice(None)] * (x.ndim-1) + [slice(-1, 0)]
            x[last_nyquist_ind].imag = 0.
        
        return irfftn(x, s=mode_sizes, dim=fft_dims, norm=norm)



