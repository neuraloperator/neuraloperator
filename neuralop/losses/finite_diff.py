import torch
"""
finite_diff.py implements utilities for computing derivatives via finite-difference method
"""

#Set fix{x,y,z}_bnd if function is non-periodic in {x,y,z} direction
#x: (*, s)
#y: (*, s)
def central_diff_1d(x, h, fix_x_bnd=False):
    """central_diff_1d computes the first spatial derivative
    of x using central finite-difference 

    Parameters
    ----------
    x : torch.Tensor
        input data on a regular 1d grid, such that
        x[i] = f(x_i)
    h : float
        discretization size of input x
    fix_x_bnd : bool, optional
        whether to average boundary and second-outermost 
        derivative values, by default False

    Returns
    -------
    dx
        output tensor of df(x)/dx at each point
    """
    dx = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h)

    if fix_x_bnd:
        dx[...,0] = (x[...,1] - x[...,0])/h
        dx[...,-1] = (x[...,-1] - x[...,-2])/h
    
    return dx

#x: (*, s1, s2)
#y: (*, s1, s2)
def central_diff_2d(x, h, fix_x_bnd=False, fix_y_bnd=False):
    """central_diff_2d computes derivatives 
    df(x,y)/dx and df(x,y)/dy for f(x,y) defined 
    on a regular 2d grid using finite-difference

    Parameters
    ----------
    x : torch.Tensor
        input function defined x[:,i,j] = f(x_i, y_j)
    h : float or list
        discretization size of grid for each dimension
    fix_x_bnd : bool, optional
        whether to fix dx on the x boundaries, by default False
    fix_y_bnd : bool, optional
        whether to fix dy on the y boundaries, by default False

    Returns
    -------
    dx, dy
        tuple such that dx[:, i,j]= df(x_i,y_j)/dx
        and dy[:, i,j]= df(x_i,y_j)/dy
    """
    if isinstance(h, float):
        h = [h, h]

    dx = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2))/(2.0*h[0])
    dy = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h[1])

    if fix_x_bnd:
        dx[...,0,:] = (x[...,1,:] - x[...,0,:])/h[0]
        dx[...,-1,:] = (x[...,-1,:] - x[...,-2,:])/h[0]
    
    if fix_y_bnd:
        dy[...,:,0] = (x[...,:,1] - x[...,:,0])/h[1]
        dy[...,:,-1] = (x[...,:,-1] - x[...,:,-2])/h[1]
        
    return dx, dy

#x: (*, s1, s2, s3)
#y: (*, s1, s2, s3)
def central_diff_3d(x, h, fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
    """central_diff_3d computes derivatives 
    df(x,y,z)/dx and df(x,y,z)/dy for f(x,y,z) defined 
    on a regular 2d grid using finite-difference

    Parameters
    ----------
    x : torch.Tensor
        input function defined x[:,i,j,k] = f(x_i, y_j,z_k)
    h : float or list
        discretization size of grid for each dimension
    fix_x_bnd : bool, optional
        whether to fix dx on the x boundaries, by default False
    fix_y_bnd : bool, optional
        whether to fix dy on the y boundaries, by default False
    fix_z_bnd : bool, optional
        whether to fix dz on the z boundaries, by default False

    Returns
    -------
    dx, dy, dz
        tuple such that dx[:, i,j,k]= df(x_i,y_j,z_k)/dx
        and dy[:, i,j,k]= df(x_i,y_j,z_k)/dy
        and dz[:, i,j,k]= df(x_i,y_j,z_k)/dz
    """
    if isinstance(h, float):
        h = [h, h, h]

    dx = (torch.roll(x, -1, dims=-3) - torch.roll(x, 1, dims=-3))/(2.0*h[0])
    dy = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2))/(2.0*h[1])
    dz = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h[2])

    if fix_x_bnd:
        dx[...,0,:,:] = (x[...,1,:,:] - x[...,0,:,:])/h[0]
        dx[...,-1,:,:] = (x[...,-1,:,:] - x[...,-2,:,:])/h[0]
    
    if fix_y_bnd:
        dy[...,:,0,:] = (x[...,:,1,:] - x[...,:,0,:])/h[1]
        dy[...,:,-1,:] = (x[...,:,-1,:] - x[...,:,-2,:])/h[1]
    
    if fix_z_bnd:
        dz[...,:,:,0] = (x[...,:,:,1] - x[...,:,:,0])/h[2]
        dz[...,:,:,-1] = (x[...,:,:,-1] - x[...,:,:,-2])/h[2]
        
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
