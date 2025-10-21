import torch
import numpy as np
import torch.nn.functional as F


def spectral_projection_divergence_free(u, domain_size, constraint_modes):
    """Apply spectral projection layer to make a velocity field divergence-free.
    
    Parameters
    ----------
    u : torch.Tensor
        Input velocity field [batch, 2, height, width] where the last two
        dimensions represent the 2D velocity components (u_x, u_y)
    domain_size : float or tuple of float
        Physical domain size. If float, assumes square domain with same size for height and width.
        If tuple, should be (height_size, width_size) for non-square domains.
    constraint_modes : tuple
        Number of modes to use for constraint resolution (height_modes, width_modes).
        If larger than input dimensions, they are truncated to the input dimensions.
    
    Returns
    -------
    torch.Tensor
        Divergence-free projected velocity field with [batch, 2, height, width] shape.
        The output maintains the same shape as the input while satisfying ∇·u = 0.
    
    

    Mathematical formulation:
    -------------------------
    
    This method implements a Helmholtz-Hodge projection in the spectral domain
    to project velocity fields onto the space of divergence-free functions.
    
    The Helmholtz-Hodge projection is given by:
        û_proj = û - (k·û)/|k|² * k
    where û is the Fourier transform of the velocity field, k is the wavenumber vector, 
    and û_proj is the projected divergence-free field.
    
    The projection enforces ∇·u = 0 in the spectral domain by removing the
    irrotational component of the velocity field.
    
        
    Periodicity Assumption:
    -----------------------
    Just like most spectral methods, this spectral projection assumes the given and 
    desired velocity fields are periodic. 
    
    If the velocity fields are not periodic, 
    one way to apply this spectral projection is to proceed as follows:
      1) extend the velocity fields to periodic velocity fields on a larger domain, using 
          Fourier continuation for instance (see neuralop.layers.fourier_continuation) 
      2) apply the spectral projection to the periodic fields on the extended domain.
      3) truncate the result back to the original domain.
      
    This is similar to how Fourier/spectral differentiation can still be performed 
    on non-periodic fields (see neuralop.losses.fourier_diff for implementation details).
    

    References:
    -----------
    
    The method is based on the spectral projection approach described in:
    
        [1] Towards Enforcing Hard Physics Constraints in Operator Learning Frameworks
        V. Duruisseaux, M. Liu-Schiaffini, J. Berner, and A. Anandkumar
        ICML 2024 AI for Science Workshop
        https://openreview.net/pdf?id=Zvxm14Rd1F
        
        [2] Enforcing physical constraints in CNNs through differentiable PDE layer. 
        C. M. Jiang, K. Kashinath, P. Prabhat, and P. Marcus
        ICLR 2020 Workshop on Integration of Deep Neural Models and Differential Equations, 2020.
        https://openreview.net/pdf?id=q2noHUqMkK
    
    
    """
    
    device = u.device
    dtype = u.dtype
    batch_size, channels, height, width = u.shape
    
    # Get domain size
    if isinstance(domain_size, (int, float)):
        # Square domain: use same size for both dimensions
        domain_height = domain_width = float(domain_size)
    elif isinstance(domain_size, (tuple, list)) and len(domain_size) == 2:
        # Non-square domain: separate sizes for height and width
        domain_height, domain_width = float(domain_size[0]), float(domain_size[1])
    
    # Ensure constraint modes do not exceed input dimensions
    constraint_modes = (min(constraint_modes[0], height), min(constraint_modes[1], width))
    
    # 2D FFT over height and width
    u_ft = torch.fft.fftn(u, dim=(2, 3))  
    
    # Extract lower modes for both dimensions where the constraint is applied
    if (height != constraint_modes[0]) or (width != constraint_modes[1]):
        u_ft = torch.fft.fftshift(u_ft, dim=(2, 3))
        u_ft = u_ft[:, :, 
                    (height - constraint_modes[0])//2 : (height - constraint_modes[0])//2 + constraint_modes[0],
                    (width - constraint_modes[1])//2 : (width - constraint_modes[1])//2 + constraint_modes[1]]
        u_ft = torch.fft.ifftshift(u_ft, dim=(2, 3))
    
    # Set up wavenumber grids for spectral operations
    ky = 2*np.pi * torch.fft.fftfreq(constraint_modes[0], d=domain_height/constraint_modes[0]).to(dtype).to(device)
    kx = 2*np.pi * torch.fft.fftfreq(constraint_modes[1], d=domain_width/constraint_modes[1]).to(dtype).to(device)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij' if constraint_modes[0] == constraint_modes[1] else 'xy')
    
    
    # Apply Helmholtz-Hodge projection: û_proj = û - (k·û)/|k|² * k
    k_dot_u = KX * u_ft[:, 0, :, :] + KY * u_ft[:, 1, :, :]
    k_squared = KX**2 + KY**2 + 1e-8    # Add small epsilon to avoid division by zero
    
    projected_u_ft = u_ft - (k_dot_u / k_squared).unsqueeze(1) * torch.stack([KX, KY], dim=0).unsqueeze(0)
    
    # Handle zero mode explicitly
    projected_u_ft[:, :, 0, 0] = 0.0
    
    
    # Pad zeros back to full resolution if needed
    if height != constraint_modes[0] or width != constraint_modes[1]:
        projected_u_ft = torch.fft.fftshift(projected_u_ft)
        pad_h = height - constraint_modes[0]
        pad_w = width - constraint_modes[1]
        projected_u_ft = F.pad(projected_u_ft, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        projected_u_ft = torch.fft.ifftshift(projected_u_ft)
    
    # Transform back to physical space
    return torch.fft.ifftn(projected_u_ft, dim=(2, 3)).real
