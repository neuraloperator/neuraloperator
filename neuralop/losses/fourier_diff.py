from neuralop.layers.fourier_continuation import FCLegendre
import torch
import warnings

def fourier_derivative_1d(u, order=1, L=2*torch.pi, use_FC=False, FC_n=4, FC_d=40, FC_one_sided=False):
    """
    Compute the 1D Fourier derivative of a given tensor.
    Use Fourier continuation to extend the signal if it is non-periodic. 
    Use with care, as Fourier continuation and Fourier derivatives are not always stable.

    Parameters
    ----------
    u : torch.Tensor
        Input tensor. The derivative will be computed along the last dimension.
    order : int, optional
        Order of the derivative. Defaults to 1.
    L : float, optional
        Length of the domain considered. Defaults to 2*pi.
    use_FC : bool, optional
        Whether to use Fourier continuation. Use for non-periodic functions. Defaults to False.
    FC_n : int, optional
        Degree of the Fourier continuation. Defaults to 4.
    FC_d : int, optional
        Number of points to add using the Fourier continuation layer. Defaults to 40.
    FC_one_sided : bool, optional
        Whether to only add points on one side, or add an equal number of points on both sides. Defaults to False.

    Returns
    -------
    torch.Tensor
        The derivative of the input tensor.
    """
    
    # Extend signal using Fourier continuation if specified
    if use_FC:
        FC = FCLegendre(n=FC_n, d=FC_d).to(u.device)
        u = FC(u, dim=1, one_sided=FC_one_sided)
        L = L *  (u.shape[-1] + FC_d) / u.shape[-1]     # Define extended length
    else:
        warnings.warn("Consider using Fourier continuation if the input is not periodic (use_FC=True).", category=UserWarning)


    nx = u.size(-1)    
    u_h = torch.fft.rfft(u, dim=-1) 
    k_x = torch.fft.rfftfreq(nx, d=1/nx, device=u_h.device).view(*([1] * (u_h.dim() - 1)), u_h.size(-1))
    
    # Fourier differentiation
    derivative_u_h = (1j * k_x * 2*torch.pi/L)**order * u_h 
    
    # Inverse Fourier transform to get the derivative in physical space
    derivative_u = torch.fft.irfft(derivative_u_h, dim=-1, n=nx) 

    # If Fourier continuation is used, crop the result to retrieve the derivative on the original interval
    if use_FC:
        if FC_one_sided:
            derivative_u = derivative_u[..., :-FC_d]
        else:
            derivative_u = derivative_u[..., FC_d//2: -FC_d//2]

    return derivative_u
