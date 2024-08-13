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

