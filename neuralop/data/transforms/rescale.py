import torch
import numpy as np
import torchvision.transforms.functional as F


def DarcyExtractBC(x, y):
    """
    Extract boundary conditions for Darcy Equation: 2D boundary value problem.
    Code adapted from paper https://arxiv.org/abs/2507.18813

    Attributes
    ----------
    x: torch.Tensor
        initial condition, a tensor of shape (batch, channels, X, Y)
    y: torch.Tensor
        boundary condition, a tensor of shape (batch, channels, X, Y)

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
    Extract boundary conditions for Burgers equation: 1D+time initial value problem.
    Code adapted from paper https://arxiv.org/abs/2507.18813

    Attributes
    ----------
    x: torch.Tensor
        initial condition, a tensor of shape (batch, channels, X, T)
    y: torch.Tensor
        boundary condition, a tensor of shape (batch, channels, X, T)

    Returns
    -------
    x: torch.Tensor
        the combined input tensor with boundary conditions added in channels 1 and 2
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
    Extract boundary conditions for Helmholtz Equation: 2D boundary value problem.
    Different from Darcy, the Helmholtz Equation is complex-valued and have two channels for real and imaginary parts.
    Code adapted from paper https://arxiv.org/abs/2507.18813

    Attributes
    ----------
    x: torch.Tensor
        initial condition, a tensor of shape (batch, channels, X, Y)
    y: torch.Tensor
        boundary condition, a tensor of shape (batch, channels, X, Y)

    Returns
    -------
    x: torch.Tensor
        the combined input tensor with boundary conditions added in channels 1,...,8
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


def GridResize(x, grid_size, mode="bilinear"):
    """
    Resize the grid values by interpolation in the last two components.
    Expected input is either of size
        batch x channel x original_size x original_size
    or
        channel x original_size x original_size
    """
    if x.ndim == 4:
        return torch.nn.functional.interpolate(
            x, size=(grid_size, grid_size), mode="bilinear", align_corners=True
        )
    elif x.ndim == 3:
        return torch.nn.functional.interpolate(
            x.unsqueeze(0),
            size=(grid_size, grid_size),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
    else:
        ValueError(
            f"Input x to GridResize must be a tensor with either 3 or 4 dimensions! {x.ndims=}, {x.shape=}"
        )


# data augmentation routines
class GridResizing:
    def __init__(self, grid_size, ExtractBC=DarcyExtractBC):
        self.grid_size = grid_size

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        x = GridResize(x, self.grid_size)
        y = GridResize(y, self.grid_size)
        x = ExtractBC(x, y)
        return x, y


class RandomFlip:
    def __init__(self, p=0.5, ExtractBC=DarcyExtractBC):
        self.p = p
        self.ExtractBC = ExtractBC

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        if torch.rand(1) < self.p:
            x, y = F.hflip(x), F.hflip(y)
        if torch.rand(1) < self.p:
            x, y = F.vflip(x), F.vflip(y)
        if torch.rand(1) < self.p:
            x, y = x.transpose(-1, -2), y.transpose(-1, -2)
        x = self.ExtractBC(x, y)
        return x, y


class RandomRotation:
    def __init__(self, p=0.5, ExtractBC=DarcyExtractBC):
        raise NotImplementedError

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        if torch.rand(1) < self.p:
            x, y = x.transpose(-1, -2), y.transpose(-1, -2)
        x = ExtractBC(x, y)
        return x, y


class RandomCropResize:
    def __init__(self, p=0.5, scale_min=0.1, size_min=32):
        """
        Args:
            p (float): probability with which to apply the transformation
            scale_min (float): minimal relative scale of subdomain vs. global domain
            size_min (int): minimal size in terms of grid points
        """
        self.p = p
        self.scale_min = scale_min
        self.size_min = size_min
        self.bbox = None
        #
        assert (
            self.scale_min <= 1.0
        ), f"Scaling factor can at most be 1.0, got {scale_min=}"

    def get_params(self, x, y, re=1, rate=None):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            x,y (tensor): Input and corresponding output (assumed having same last two dimensions).

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = x.shape[-2], x.shape[-1]
        assert width == height, "Only allowing width==height for now!."
        assert (
            width >= self.size_min
        ), f"Cropping only allowed if (width,height)>={self.size_min}. Got {width=}, {height=}"
        #
        size_min = max(self.size_min, int(round(self.scale_min * width)))
        for _ in range(10):
            if rate == None:
                rnd = torch.rand(1)
                w = size_min + int((width - size_min) * rnd)
                h = size_min + int((height - size_min) * rnd)
            else:
                if rate < 1:
                    rate = 1 / rate
                w = int(max(size_min, width // rate))
                h = int(max(size_min, height // rate))

            # # just for sanity
            # w = min(w,width)
            # h = min(h,height)

            scale = np.sqrt((w / width) * (h / height))
            re = re * scale

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w, re

    def crop(self, x, i, j, h, w):
        return x[..., i : i + h, j : j + w]


class RandomCropResizeTime:
    def __init__(self, p=0.5, scale_min=0.1, size_min=32):
        """
        Args:
            p (float): probability with which to apply the transformation
            scale_min (float): minimal relative scale of subdomain vs. global domain
            size_min (int): minimal size in terms of grid points
        """
        self.p = p
        self.scale_min = scale_min
        self.size_min = size_min
        self.bbox = None
        #
        assert (
            self.scale_min <= 1.0
        ), f"Scaling factor can at most be 1.0, got {scale_min=}"

    def get_params(self, x, y, re=1, rate=None):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            x,y (tensor): Input and corresponding output (assumed having same last two dimensions).

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        T, S = x.shape[-2], x.shape[-1]
        assert (
            T >= self.size_min
        ), f"Cropping only allowed if (width,height)>={self.size_min}. Got {width=}, {height=}"

        size_min = max(self.size_min, int(round(self.scale_min * T)))

        if rate == None:
            rnd = torch.rand(1)
            t = size_min + int((T - size_min) * rnd)
            rate = T / t
        else:
            if rate < 1:
                rate = 1 / rate

        t = int(max(size_min, T // rate))
        s = int(max(size_min, S // rate))

        scale = s / S
        re = re * scale

        i = torch.randint(0, T - t + 1, size=(1,)).item()
        j = torch.randint(0, S - s + 1, size=(1,)).item()

        return i, j, t, s, re

    def crop(self, x, i, j, h, w):
        return x[..., i : i + h, j : j + w]


class RandomCropResizeTimeAR:
    def __init__(self, p=0.5, scale_min=0.1, size_min=32):
        """
        Args:
            p (float): probability with which to apply the transformation
            scale_min (float): minimal relative scale of subdomain vs. global domain
            size_min (int): minimal size in terms of grid points
        """
        self.p = p
        self.scale_min = scale_min
        self.size_min = size_min
        self.bbox = None
        #
        assert (
            self.scale_min <= 1.0
        ), f"Scaling factor can at most be 1.0, got {scale_min=}"

    def get_params(self, x, y, re=1, rate=None):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            x,y (tensor): Input and corresponding output (assumed having same last two dimensions).

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """

        width, height = x.shape[-2], x.shape[-1]
        assert width == height, "Only allowing width==height for now!."
        assert (
            width >= self.size_min
        ), f"Cropping only allowed if (width,height)>={self.size_min}. Got {width=}, {height=}"
        #
        size_min = max(self.size_min, int(round(self.scale_min * width)))
        for _ in range(10):
            if rate == None:
                rnd = torch.rand(1)
                w = size_min + int((width - size_min) * rnd)
                h = size_min + int((height - size_min) * rnd)
            else:
                if rate < 1:
                    rate = 1 / rate
                w = int(max(size_min, width // rate))
                h = int(max(size_min, height // rate))

            scale = np.sqrt((w / width) * (h / height))
            re = re * scale**2

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                #
                return i, j, h, w, re

    def crop(self, x, i, j, h, w):
        x = x[..., i : i + h, j : j + w]
        return x
