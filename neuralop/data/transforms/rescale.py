import torch


def DarcyExtractBC(x, y):
    """
    Extract boundary conditions for Darcy boundary value problem.
    Code adapted from paper https://arxiv.org/abs/2507.18813

    Attributes
    ----------
    x: torch.Tensor
        initial condition, a tensor of shape (batch, channels, height, width)
    y: torch.Tensor
        boundary condition, a tensor of shape (batch, channels, height, width)

    Returns
    -------
    x: torch.Tensor
        the combined input tensor with boundary conditions added in channels 1,...,4
    """
    unsqueeze_x = x.ndim == 3  # check whether batch dimension is missing
    unsqueeze_y = y.ndim == 3

    # add batch dimension if there was no batch dimension
    if unsqueeze_x:
        x = x.unsqueeze(0)
    if unsqueeze_y:
        y = y.unsqueeze(0)

    # extract BC
    Nx, Ny = y.shape[-2], y.shape[-1]
    x[:, 1, :, :] = y[:, 0, 0, :].unsqueeze(1).repeat(1, Nx, 1)  # left boundary
    x[:, 2, :, :] = y[:, 0, :, 0].unsqueeze(2).repeat(1, 1, Ny)  # lower boundary
    x[:, 3, :, :] = y[:, 0, -1, :].unsqueeze(1).repeat(1, Nx, 1)  # right boundary
    x[:, 4, :, :] = y[:, 0, :, -1].unsqueeze(2).repeat(1, 1, Ny)  # upper boundary

    # undo adding of batch dimension if there was no batch dimension
    if unsqueeze_x:
        x = x[0]
    if unsqueeze_y:
        y = y[0]

    return x


def BurgersExtractBC(y):
    """
    Extract boundary conditions for Darcy boundary value problem.

    Attributes
    ----------
    x: torch.Tensor
        initial condition, a tensor of shape (batch, channels, height, width)
    y: torch.Tensor
        boundary condition, a tensor of shape (batch, channels, height, width)

    Returns
    -------
    x: torch.Tensor
        the combined input tensor with boundary conditions added in channels 1,...,4
    """

    if y.ndim > 3:
        y = y.squeeze()
    T, S = y.shape[1], y.shape[2]

    # extract BC
    boundary0 = y[..., 0]
    boundary1 = y[..., -1]
    boundary = torch.stack([boundary0, boundary1], dim=1)  # (batch, 2, T)
    boundary = boundary.unsqueeze(-1).repeat(1, 1, 1, S)  # (batch, 2, T, S)

    x = y[:, 0, :]  # x (batch, x)
    x = x.reshape(-1, 1, 1, S).repeat(1, 1, T, 1)  # x (batch, 1, T, x)
    x = torch.cat([x, boundary], dim=1)  # x (batch, 1+2, T, x)
    return x


def HelmholtzExtractBC(x, y):
    """
    Extract boundary conditions from y and add them to channels 1,...,4 in x.
    """
    unsqueeze_x = x.ndim == 3  # check whether batch dimension is missing
    unsqueeze_y = y.ndim == 3

    # add batch dimension
    if unsqueeze_x:
        x = x.unsqueeze(0)
    if unsqueeze_y:
        y = y.unsqueeze(0)

    # extract BC
    Nx, Ny = y.shape[-2], y.shape[-1]
    x[:, 1:3, :, :] = y[:, :, 0, :].unsqueeze(-2).repeat(1, 1, Nx, 1)  # left boundary
    x[:, 3:5, :, :] = y[:, :, :, 0].unsqueeze(-1).repeat(1, 1, 1, Ny)  # lower boundary
    x[:, 5:7, :, :] = y[:, :, -1, :].unsqueeze(-2).repeat(1, 1, Nx, 1)  # right boundary
    x[:, 7:9, :, :] = y[:, :, :, -1].unsqueeze(-1).repeat(1, 1, 1, Ny)  # upper boundary

    # undo adding of batch dimension
    if unsqueeze_x:
        x = x[0]
    if unsqueeze_y:
        y = y[0]

    return x
