from neuralop.layers.fourier_continuation import FCLegendre, FCGram
import torch
import warnings


def fourier_derivative_1d(u, order=1, L=2*torch.pi, use_FC=False, FC_d=4, FC_n_additional_pts=50, FC_one_sided=False, low_pass_filter_ratio=None):
    """
    Compute the 1D Fourier derivative of a given tensor.
    Use with care, as Fourier continuation and Fourier derivatives are not always stable.
    
    FC derivatives are more stable when the original signal is nearly periodic 
        (i.e., values and slopes at boundaries match or are close).
        and when the signal is smooth and has no sharp jumps/discontinuities at the boundaries.
    
    Use Fourier continuation to extend the signal if it is non-periodic. 
    
    Unstable behavior can occur if the signal has strong discontinuities or non-matching derivatives at boundaries,
        and if the continuation introduces artificial oscillations (Gibbs phenomenon).

    Signs of instability include boundary artifacts, high-frequency ringing, or growing errors with higher resolution.
    
    
    Parameters
    ----------
    u : torch.Tensor
        Input tensor. The derivative will be computed along the last dimension.
    order : int, optional
        Order of the derivative, by default 1
    L : float, optional
        Length of the domain considered, by default 2*pi
    use_FC : str, optional
        Whether to use Fourier continuation. Use for non-periodic functions.
        Options: None, 'Legendre', 'Gram', by default False
    FC_d : int, optional
        'Degree' of the Fourier continuation, by default 4
    FC_n_additional_pts : int, optional
        Number of points to add using the Fourier continuation layer, by default 50
        For FC-Gram continuation, it is usually not necessary to change this parameter. 
        This has a bigger effect on FC-Legendre continuation.
    FC_one_sided : bool, optional
        Whether to only add points on one side, or add an equal number of points 
        on both sides, by default False
    low_pass_filter_ratio : float, optional
        If not None, apply a low-pass filter to the Fourier coefficients. 
        Can help reduce artificial oscillations. 1.0 means no filtering, 
        0.5 means keep half of the coefficients, etc., by default None

    Returns
    -------
    torch.Tensor
        The derivative of the input tensor.
        
    Notes
    -----
    When using Fourier continuation, the function automatically adjusts the 
    domain length L to account for the extended signal. The result is cropped 
    back to the original interval size.
    
    Warnings
    --------
    Consider using Fourier continuation if the input is not periodic (use_FC=True).
    Fourier continuation and Fourier derivatives can be numerically unstable
    for certain functions and parameter combinations.
    """
    
    # Extend signal using Fourier continuation if specified
    if use_FC=='Legendre':
        FC = FCLegendre(d=FC_d, n_additional_pts=FC_n_additional_pts).to(u.device)
        u = FC(u, dim=1, one_sided=FC_one_sided)
        L = L *  (u.shape[-1] + FC_n_additional_pts) / u.shape[-1]     # Define extended length
    elif use_FC=='Gram':
        FC = FCGram(d=FC_d, n_additional_pts=FC_n_additional_pts).to(u.device)
        u = FC(u, dim=1, one_sided=FC_one_sided)
        L = L *  (u.shape[-1] + FC_n_additional_pts) / u.shape[-1]    
    else:
        warnings.warn("Consider using Fourier continuation if the input is not periodic (use_FC=True).", category=UserWarning)

    nx = u.size(-1)    
    u_h = torch.fft.rfft(u, dim=-1) 
    k_x = torch.fft.rfftfreq(nx, d=1/nx, device=u_h.device).view(*([1] * (u_h.dim() - 1)), u_h.size(-1))
    
    if low_pass_filter_ratio is not None:
        # Apply a low-pass filter to the Fourier coefficients
        cutoff = int(u_h.shape[-1] * low_pass_filter_ratio)
        u_h[..., cutoff:] = 0
    
    # Fourier differentiation
    derivative_u_h = (1j * k_x * 2*torch.pi/L)**order * u_h 
    
    # Inverse Fourier transform to get the derivative in physical space
    derivative_u = torch.fft.irfft(derivative_u_h, dim=-1, n=nx) 

    # If Fourier continuation is used, crop the result to retrieve the derivative on the original interval
    if use_FC:
        if FC_one_sided:
            derivative_u = derivative_u[..., :-FC_n_additional_pts]
        else:
            derivative_u = derivative_u[..., FC_n_additional_pts//2: -FC_n_additional_pts//2]

    return derivative_u
