import torch
"""
finite_diff.py implements utilities for computing derivatives via finite-difference method
"""

#x: (*, s)
#y: (*, s)
def central_diff_1d(x, h, periodic_in_x=True):
    """central_diff_1d computes the first spatial derivative
    of x using central finite-difference 

    This function computes df/dx using the central difference formula:
    df/dx \approx (f(x+h) - f(x-h)) / (2h)
    
    For periodic domains (periodic_in_x=True), the function uses torch.roll to handle
    boundary wrapping, treating the domain as periodic.
    
    For non-periodic domains (periodic_in_x=False), the function uses forward differences
    at the left boundary and backward differences at the right boundary to avoid
    accessing points outside the domain.

    Parameters
    ----------
    x : torch.Tensor
        input data on a regular 1d grid, such that
        x[i] = f(x_i)
    h : float
        discretization size of input x
    periodic_in_x : bool, optional
        whether to use periodic boundary conditions:
        - True: periodic domain (default)
        - False: non-periodic domain with forward/backward differences at boundaries
        by default True

    Returns
    -------
    dx : torch.Tensor
        output tensor of df(x)/dx at each point
    """
    if periodic_in_x:
        # Periodic case: use torch.roll for boundary wrapping
        dx = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1)) / (2.0 * h)
    else:
        # Non-periodic case: handle boundaries separately
        dx = torch.zeros_like(x)
        
        # Interior points: central difference
        dx[..., 1:-1] = (x[..., 2:] - x[..., :-2]) / (2.0 * h)
        
        # Boundary points: forward and backward differences
        dx[..., 0] = (x[..., 1] - x[..., 0]) / h
        dx[..., -1] = (x[..., -1] - x[..., -2]) / h
    
    return dx

#x: (*, s1, s2)
#y: (*, s1, s2)
def central_diff_2d(x, h, periodic_in_x=True, periodic_in_y=True):
    """central_diff_2d computes derivatives 
    df(x,y)/dx and df(x,y)/dy for f(x,y) defined 
    on a regular 2d grid using finite-difference

    This function computes partial derivatives using the central difference formula:
    df/dx \approx (f(x+h,y) - f(x-h,y)) / (2h_x)
    df/dy \approx (f(x,y+h) - f(x,y-h)) / (2h_y)
    
    For periodic dimensions (periodic_in_*=True), the function uses torch.roll to handle
    boundary wrapping, treating those dimensions as periodic.
    
    For non-periodic dimensions (periodic_in_*=False), the function uses forward differences
    at the left boundary and backward differences at the right boundary to avoid
    accessing points outside the domain.

    Parameters
    ----------
    x : torch.Tensor
        input function defined x[:,i,j] = f(x_i, y_j)
    h : float or list
        discretization size of grid for each dimension
    periodic_in_x : bool, optional
        whether to use periodic boundary conditions in x-direction:
        - True: periodic in x (default)
        - False: non-periodic in x with forward/backward differences at boundaries
        by default True
    periodic_in_y : bool, optional
        whether to use periodic boundary conditions in y-direction:
        - True: periodic in y (default)
        - False: non-periodic in y with forward/backward differences at boundaries
        by default True

    Returns
    -------
    dx, dy : tuple of torch.Tensor
        tuple such that dx[:, i,j]= df(x_i,y_j)/dx
        and dy[:, i,j]= df(x_i,y_j)/dy
    """
    if isinstance(h, float):
        h = [h, h]

    if periodic_in_x:
        # Periodic case in x-direction: use torch.roll for boundary wrapping
        dx = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2)) / (2.0 * h[0])
    else:
        # Non-periodic case in x-direction: handle boundaries separately
        dx = torch.zeros_like(x)
        
        # Interior points: central difference
        dx[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) / (2.0 * h[0])
        
        # Boundary points: forward and backward differences
        dx[..., 0, :] = (x[..., 1, :] - x[..., 0, :]) / h[0]
        dx[..., -1, :] = (x[..., -1, :] - x[..., -2, :]) / h[0]

    if periodic_in_y:
        # Periodic case in y-direction: use torch.roll for boundary wrapping
        dy = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1)) / (2.0 * h[1])
    else:
        # Non-periodic case in y-direction: handle boundaries separately
        dy = torch.zeros_like(x)
        
        # Interior points: central difference
        dy[..., :, 1:-1] = (x[..., :, 2:] - x[..., :, :-2]) / (2.0 * h[1])
        
        # Boundary points: forward and backward differences
        dy[..., :, 0] = (x[..., :, 1] - x[..., :, 0]) / h[1]
        dy[..., :, -1] = (x[..., :, -1] - x[..., :, -2]) / h[1]
        
    return dx, dy

#x: (*, s1, s2, s3)
#y: (*, s1, s2, s3)
def central_diff_3d(x, h, periodic_in_x=True, periodic_in_y=True, periodic_in_z=True):
    """central_diff_3d computes derivatives 
    df(x,y,z)/dx, df(x,y,z)/dy, and df(x,y,z)/dz for f(x,y,z) defined 
    on a regular 3d grid using finite-difference

    This function computes partial derivatives using the central difference formula:
    df/dx \approx (f(x+h,y,z) - f(x-h,y,z)) / (2h_x)
    df/dy \approx (f(x,y+h,z) - f(x,y-h,z)) / (2h_y)
    df/dz \approx (f(x,y,z+h) - f(x,y,z-h)) / (2h_z)
    
    For periodic dimensions (periodic_in_*=True), the function uses torch.roll to handle
    boundary wrapping, treating those dimensions as periodic.
    
    For non-periodic dimensions (periodic_in_*=False), the function uses forward differences
    at the left boundary and backward differences at the right boundary to avoid
    accessing points outside the domain.

    Parameters
    ----------
    x : torch.Tensor
        input function defined x[:,i,j,k] = f(x_i, y_j, z_k)
    h : float or list
        discretization size of grid for each dimension
    periodic_in_x : bool, optional
        whether to use periodic boundary conditions in x-direction:
        - True: periodic in x (default)
        - False: non-periodic in x with forward/backward differences at boundaries
        by default True
    periodic_in_y : bool, optional
        whether to use periodic boundary conditions in y-direction:
        - True: periodic in y (default)
        - False: non-periodic in y with forward/backward differences at boundaries
        by default True
    periodic_in_z : bool, optional
        whether to use periodic boundary conditions in z-direction:
        - True: periodic in z (default)
        - False: non-periodic in z with forward/backward differences at boundaries
        by default True

    Returns
    -------
    dx, dy, dz : tuple of torch.Tensor
        tuple such that dx[:, i,j,k]= df(x_i,y_j,z_k)/dx
        and dy[:, i,j,k]= df(x_i,y_j,z_k)/dy
        and dz[:, i,j,k]= df(x_i,y_j,z_k)/dz
    """
    if isinstance(h, float):
        h = [h, h, h]

    if periodic_in_x:
        # Periodic case in x-direction: use torch.roll for boundary wrapping
        dx = (torch.roll(x, -1, dims=-3) - torch.roll(x, 1, dims=-3)) / (2.0 * h[0])
    else:
        # Non-periodic case in x-direction: handle boundaries separately
        dx = torch.zeros_like(x)
        
        # Interior points: central difference
        dx[..., 1:-1, :, :] = (x[..., 2:, :, :] - x[..., :-2, :, :]) / (2.0 * h[0])
        
        # Boundary points: forward and backward differences
        dx[..., 0, :, :] = (x[..., 1, :, :] - x[..., 0, :, :]) / h[0]
        dx[..., -1, :, :] = (x[..., -1, :, :] - x[..., -2, :, :]) / h[0]

    if periodic_in_y:
        # Periodic case in y-direction: use torch.roll for boundary wrapping
        dy = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2)) / (2.0 * h[1])
    else:
        # Non-periodic case in y-direction: handle boundaries separately
        dy = torch.zeros_like(x)
        
        # Interior points: central difference
        dy[..., :, 1:-1, :] = (x[..., :, 2:, :] - x[..., :, :-2, :]) / (2.0 * h[1])
        
        # Boundary points: forward and backward differences
        dy[..., :, 0, :] = (x[..., :, 1, :] - x[..., :, 0, :]) / h[1]
        dy[..., :, -1, :] = (x[..., :, -1, :] - x[..., :, -2, :]) / h[1]

    if periodic_in_z:
        # Periodic case in z-direction: use torch.roll for boundary wrapping
        dz = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1)) / (2.0 * h[2])
    else:
        # Non-periodic case in z-direction: handle boundaries separately
        dz = torch.zeros_like(x)
        
        # Interior points: central difference
        dz[..., :, :, 1:-1] = (x[..., :, :, 2:] - x[..., :, :, :-2]) / (2.0 * h[2])
        
        # Boundary points: forward and backward differences
        dz[..., :, :, 0] = (x[..., :, :, 1] - x[..., :, :, 0]) / h[2]
        dz[..., :, :, -1] = (x[..., :, :, -1] - x[..., :, :, -2]) / h[2]
        
    return dx, dy, dz



def get_non_uniform_fd_weights(points, num_neighbors=5, derivative_indices=[0], radius=None, regularize_lstsq=False):
    """
    Compute finite difference weights for approximating the first order derivative
    on an unstructured grid of points
    Parameters:
    -----------
    points : torch tensor of shape (N, d) containing the d coordinates of the N points
    num_neighbors: int for the number of nearest neighbors to include in the stencil (including the point itself)
                    At least 3 and at most N
    derivative_indices : indices of the derivatives to compute, e.g. in 2D, [0] for x, [1] for y, [0, 1] for x and y
    radius : float, the cutoff distance to use a neighbor as radius
             Neighbors with distances beyond this value are not used (their weights are set to zero)
             Always keep at least 3 neighbors (including the point itself)
    regularize_lstsq : bool, whether to regularize the least squares system
                        Sometimes torch.linalg.lstsq(A, b).solution creates artifacts so can add regularizer
                        But regularizer can deteriorate performance when system is well-conditioned
    
    Returns:
    --------
    indices : torch tensor of shape (N, k) for the indices of k nearest neighbors (including the point itself)
    fd_weights : torch tensor of weights of shape (N, len(derivative_indices), k)
                fd_weights[i,j,m] contains the weights for the m-th nearest neighbor 
                                        in the j-th 1st order derivative for the i-th point
    """

    N = points.shape[0]
    d = points.shape[1]
    k = min(max(num_neighbors, 3), N)  

    # Get the indices of the k nearest neighbors (including the point itself)
    pairwise_distances = torch.cdist(points, points, p=2)
    distances, indices = torch.topk(pairwise_distances, k=k, dim=1, largest=False)
    
    # Get mask for neighbors within cutoff radius (and always keep at least 3)
    if radius is None:
        radius_mask = torch.ones_like(distances, dtype=torch.bool)
    else:
        radius_mask = distances <= radius
        radius_mask[:, :3] = True

    # Initialize A to 1 since first row for each point and derivative is 1
    A = torch.ones((N, d+1, k), dtype=points.dtype, device=points.device)
    # Compute coordinate differences
    for i in range(d):
        A[:, i+1, :] = points[indices, i] - points[:, i].unsqueeze(1)
    # Repeat it for each derivative to be evaluated so it becomes of shape (N, len(derivative_indices), d+1, k)
    A = A.unsqueeze(1).expand(-1, len(derivative_indices), -1, -1)

    # Zero out columns for neighbors that are not within the radius
    A = A * radius_mask.unsqueeze(1).unsqueeze(2)
    
    # Compute right hand side
    b = torch.zeros((len(derivative_indices), d+1, 1), dtype=points.dtype, device=points.device)
    for i in range(len(derivative_indices)):
        b[i, derivative_indices[i]+1] = 1
    # Repeat so it becomes (N, len(derivative_indices), d+1, 1)
    b = b.unsqueeze(0).expand(N, -1, -1, -1)

    # Solve least squares system Aw = b  
    #    sometimes torch.linalg.lstsq(A, b).solution creates artifacts so can add regularizer
    #    but regularizer can deteriorate performance when system is well-conditioned

    if regularize_lstsq:

        lambda_reg = 1e-6
        I_k = torch.eye(k, dtype=A.dtype, device=A.device).unsqueeze(0).unsqueeze(0)

        AT = A.transpose(-2, -1)
        AT_b = torch.matmul(AT, b)
        AT_A = torch.matmul(AT, A) + lambda_reg * I_k

        # Use Cholesky decomposition to accelerate torch.linalg.solve(AT_A, AT_b).squeeze(-1) 
        fd_weights = torch.cholesky_solve(AT_b, torch.linalg.cholesky(AT_A)).squeeze(-1)  

    else:
        fd_weights = torch.linalg.lstsq(A, b).solution 

    return indices, fd_weights.squeeze(-1)


def non_uniform_fd(points, values, num_neighbors=5, derivative_indices=[0], radius=None, regularize_lstsq=False):
    """
    Compute finite difference approximation of the first order derivative on an unstructured grid of points
    Parameters:
    -----------
    points : torch tensor of shape (N, d) containing the d coordinates of the N points
    values : torch tensor of shape (N) containing the values of the function at the N points
    radius : float, the cutoff distance to use a neighbor as radius
             Neighbors with distances beyond this value are not used (their weights are set to zero)
             Always keep at least 3 neighbors (including the point itself)
    num_neighbors: int for the number of nearest neighbors to include in the stencil (including the point itself)
    derivative_indices : indices of the derivatives to compute, e.g. in 2D, [0] for x, [1] for y, [0, 1] for x and y
    regularize_lstsq : bool, whether to regularize the least squares system
                        Sometimes torch.linalg.lstsq(A, b).solution creates artifacts so can add regularizer
                        But regularizer can deteriorate performance when system is well-conditioned
    
    Returns:
    --------
    derivatives: tensor of shape (len(derivative_indices), N) of derivatives
            e.g. in 2D with derivative_indices=[0, 1], derivatives[0] is df(x,y)/dx and derivatives[1] is df(x,y)/dy
    
    """

    indices, fd_weights = get_non_uniform_fd_weights(points=points, 
                                                    num_neighbors=num_neighbors, 
                                                    derivative_indices=derivative_indices,
                                                    radius=radius,
                                                    regularize_lstsq=regularize_lstsq)

    derivatives = torch.einsum('nij,nj->in', fd_weights, values[indices])

    return derivatives
