import numpy as np
import torch
import torch.nn.functional as F


def spectral_projection_divergence_free(u, domain_size, constraint_modes):
    """Apply spectral projection layer to make a velocity field divergence-free.
    
    Parameters
    ----------
    u : torch.Tensor
        Input velocity field of shape ``[batch, 2, height, width]``, where channel 0
        and channel 1 represent the velocity components along the ``height`` (dim -2)
        and ``width`` (dim -1) spatial axes, matching :meth:`FourierDiff.divergence`.
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

    Notes
    -----
    **Mathematical formulation:** Helmholtz-Hodge projection in spectral domain:
    û_proj = û - (k·û)/|k|² * k for nonzero wavenumbers ``k != 0``, while preserving
    the zero-wavenumber (mean/DC) mode. Enforces ∇·u = 0 by removing the irrotational component.
    **Periodicity:** Velocity fields are assumed periodic. For non-periodic fields, extend
    via Fourier continuation (neuralop.layers.fourier_continuation), project, then truncate.

    References
    ----------
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
    _, _, height, width = u.shape
    
    # Get domain size
    if isinstance(domain_size, (int, float)):
        # Square domain: use same size for both dimensions
        domain_height = domain_width = float(domain_size)
    elif isinstance(domain_size, (tuple, list)) and len(domain_size) == 2:
        # Non-square domain: separate sizes for height and width
        domain_height, domain_width = float(domain_size[0]), float(domain_size[1])
    else:
        raise ValueError(f"domain_size must be a float or 2-tuple/list, got {domain_size!r}")
    
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
    
    # Set up wavenumber grids along (height, width) regardless of aspect ratio
    k_height = (
        2
        * np.pi
        * torch.fft.fftfreq(constraint_modes[0], d=domain_height / constraint_modes[0], device=device, dtype=dtype)
    )
    k_width = (
        2
        * np.pi
        * torch.fft.fftfreq(constraint_modes[1], d=domain_width / constraint_modes[1], device=device, dtype=dtype)
    )
    K_height, K_width = torch.meshgrid(k_height, k_width, indexing="ij")
    
    # Apply Helmholtz-Hodge projection on nonzero modes: û_proj = û - (k·û)/|k|² * k
    # Leaving k = 0 untouched preserves the spatial mean (constant flows already satisfy ∇·u = 0).
    k_dot_u = K_height * u_ft[:, 0, :, :] + K_width * u_ft[:, 1, :, :]
    k_squared = K_height**2 + K_width**2
    nonzero_k = k_squared > 0
    safe_k_squared = torch.where(nonzero_k, k_squared, torch.ones_like(k_squared))
    proj_coeff = torch.where(nonzero_k, k_dot_u / safe_k_squared, torch.zeros_like(k_dot_u))
    
    k_grid = torch.stack([K_height, K_width], dim=0).unsqueeze(0)
    projected_u_ft = u_ft - proj_coeff.unsqueeze(1) * k_grid
    
    # Pad zeros back to full resolution if needed
    if height != constraint_modes[0] or width != constraint_modes[1]:
        projected_u_ft = torch.fft.fftshift(projected_u_ft, dim=(2, 3))
        pad_h = height - constraint_modes[0]
        pad_w = width - constraint_modes[1]
        projected_u_ft = F.pad(projected_u_ft, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        projected_u_ft = torch.fft.ifftshift(projected_u_ft, dim=(2, 3))
    
    # Transform back to physical space
    return torch.fft.ifftn(projected_u_ft, dim=(2, 3)).real
