import math

import torch
from torch import nn

import neuralop.mpu.comm as comm
from neuralop.mpu.mappings import (
    gather_from_model_parallel_region,
    scatter_to_model_parallel_region,
)

class MultigridPatching2D(nn.Module):
    """
    MultigridPatching2D wraps a model in multi-grid domain decomposition and patching.
    """
    def __init__(
        self,
        model: nn.Module,
        levels: int=0,
        padding_fraction: float=0,
        use_distributed: bool=False,
        stitching: bool=True,
    ):
        """Wrap a model in MGPatching. If computation is split into distributed
        data or model parallel, adds parameter hooks to account for scattering patches across
        multiple processes.

        Parameters
        ----------
        model : nn.Module
            model to wrap 
        levels : int, optional
            number of levels of patching to use, by default 0
        padding_fraction : float, optional
            fraction of input size to add as padding around patches, by default 0
        use_distributed : bool, optional
            whether patching is performed in distributed mode, by default False
        stitching : bool, optional
            whether patches are to be stitched back together
            in spatial dimensions during training, by default True.
            Stitching is always performed during evaluation.
        """

        super().__init__()

        self.levels = levels

        if isinstance(padding_fraction, (float, int)):
            padding_fraction = [padding_fraction, padding_fraction]
        self.padding_fraction = padding_fraction

        n_patches = 2**levels
        if isinstance(n_patches, int):
            n_patches = [n_patches, n_patches]
        self.n_patches = n_patches

        self.model = model

        self.use_distributed = use_distributed
        self.stitching = stitching

        if levels > 0:
            print(
                "MGPatching("
                f"n_patches={self.n_patches}, "
                f"padding_fraction={self.padding_fraction}, "
                f"levels={self.levels}, "
                f"use_distributed={use_distributed}, "
                f"stitching={stitching})"
            )

        # If distributed patches are stiched, re-scale gradients to revert DDP averaging
        if self.use_distributed and self.stitching:
            for param in model.parameters():
                param.register_hook(
                    lambda grad: grad * float(comm.get_model_parallel_size())
                )

    def patch(self, x, y):
        """use multi-grid domain decomposition to split `x` and `y` into patches.
        If in a distributed scheme, scatters patches across processes.

        Parameters
        ----------
        x : torch.tensor
            model input function
        y : _type_
            model output function
        """
        # if not stitching in single-device, create patches for y
        if not self.stitching:
            y = make_patches(y, n=self.n_patches, p=0)
        # If not stitching, scatter truth, otherwise keep on every GPU
        if self.use_distributed:
            y = scatter_to_model_parallel_region(y, 0)

        # Create padded patches in batch dimension (identity if levels=0)
        x = self._make_mg_patches(x)
        # Split data across processes
        if self.use_distributed:
            x = scatter_to_model_parallel_region(x, 0)
        return x, y

    def unpatch(self, x, y, evaluation=False):
        """unpatch tensors created by `self.patch`. Stitch patches together if in
        evaluation mode, or if stitching is applied. 

        Parameters
        ----------
        x : torch.tensor
            tensor with patching structure created patching input `x`. May be
            either inputs `x` or raw model outputs (same shape/patching structure)
            Shape (b * n^2, c, h / n + 2 * pad_h, w / n + 2 * pad_w)
        y : torch.tensor
            tensor of patched ground-truth `y`. 
            Shape (b * n^2, c, h / n, w / n) or (b, c, h, w) when not stitched
        evaluation : bool, optional
            whether in evaluation mode, by default False.
            If True, `x` and `y` are both evaluated after stitching, 
            regardless of other settings. 
        """
        # Remove padding in the output
        if self.padding_height > 0 or self.padding_width > 0:
            x = self._unpad(x)

        # Gather patches if they are to be stitched back together
        if self.use_distributed and self.stitching:
            x = gather_from_model_parallel_region(x, dim=0)

        # Stich patches or patch the truth if output left unstitched
        if self.stitching or evaluation:
            x = self._stitch(x)
        
        # if x is not stitched during training, y is patched
        # re-stitch y during evaluation only
        if evaluation and not self.stitching:
            y = self._stitch(y)

        return x, y

    def _stitch(self, x):
        """Stitch back together multi-grid patches created by `self._make_mg_patches`.

        Small patches are collated along the batch dimension as different inputs. Unroll
        the batch dimension and stick all patches from the same input back together in their
        proper locations. 

        For an input shape (n * n * n, c, h / n, w / n),
        produces an output tensor of shape (b, c, h, w)


        Parameters
        ----------
        x : torch.tensor
            input tensor, split into patches and collated along batch dim
            shape (batch * n^2, c, h / n, w / n)
            
        """

        # Only 1D and 2D supported
        assert x.ndim == 4, f"Only 2D patch supported but got input with {x.ndim} dims."

        if self.n_patches[0] <= 1 and self.n_patches[1] <= 1:
            return x

        # Size with padding removed
        size = x.size()

        # if self.mode == "batch-wise":
        B = size[0] // (self.n_patches[0] * self.n_patches[1])
        W = size[3] * self.n_patches[1]

        C = size[1]
        H = size[2] * self.n_patches[0]

        # Reshape
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(B, self.n_patches[0], self.n_patches[1], size[3], size[2], C)
        x = x.permute(0, 5, 1, 4, 2, 3)
        x = x.reshape(B, C, H, W)

        return x

    def _make_mg_patches(self, x):
        """
        Split a tensor `x` into multi-grid patches. The number of
        patches into which `x` is split is determined by `2 ** self.levels`.

        Steps:

        1. Optionally apply padding if self.padding_fraction > 0

        2. Split each element in the batch into `2**self.levels` patches of equal size

        3. For each level L in `self.levels`, subsample all patches by a factor of `2**L`

        4. Pad the subsampled patches so all level tensors are the same size

        5. Concatenate the patches along the channel dim and return

        Parameters
        ----------
        x : torch.tensor
            input tensor to split into patches
            shape (batch, c, h, w)
        """
        levels = self.levels
        if levels <= 0:
            return x

        _, _, height, width = x.shape
        padding = [
            int(round(v))
            for v in [
                height * self.padding_fraction[0],
                width * self.padding_fraction[1],
            ]
        ]
        self.padding_height = padding[0]
        self.padding_width = padding[1]

        patched = make_patches(x, n=2**self.levels, p=padding)
        s1_patched = patched.size(-2) - 2 * padding[0]
        s2_patched = patched.size(-1) - 2 * padding[1]

        for level in range(1, levels + 1):
            sub_sample = 2**level
            s1_stride = s1_patched // sub_sample
            s2_stride = s2_patched // sub_sample

            x_sub = x[:, :, ::sub_sample, ::sub_sample]

            s2_pad = (
                math.ceil(
                    (s2_patched + (2**levels - 1) * s2_stride - x_sub.size(-1)) / 2.0
                )
                + padding[1]
            )
            s1_pad = (
                math.ceil(
                    (s1_patched + (2**levels - 1) * s1_stride - x_sub.size(-2)) / 2.0
                )
                + padding[0]
            )

            if s2_pad > x_sub.size(-1):
                diff = s2_pad - x_sub.size(-1)
                x_sub = torch.nn.functional.pad(
                    x_sub, pad=[x_sub.size(-1), x_sub.size(-1), 0, 0], mode="circular"
                )
                x_sub = torch.nn.functional.pad(
                    x_sub, pad=[diff, diff, 0, 0], mode="circular"
                )
            else:
                x_sub = torch.nn.functional.pad(
                    x_sub, pad=[s2_pad, s2_pad, 0, 0], mode="circular"
                )

            if s1_pad > x_sub.size(-2):
                diff = s1_pad - x_sub.size(-2)
                x_sub = torch.nn.functional.pad(
                    x_sub, pad=[0, 0, x_sub.size(-2), x_sub.size(-2)], mode="circular"
                )
                x_sub = torch.nn.functional.pad(
                    x_sub, pad=[0, 0, diff, diff], mode="circular"
                )
            else:
                x_sub = torch.nn.functional.pad(
                    x_sub, pad=[0, 0, s1_pad, s1_pad], mode="circular"
                )

            x_sub = x_sub.unfold(-1, s2_patched + 2 * padding[1], s2_stride)
            x_sub = x_sub.unfold(-3, s1_patched + 2 * padding[0], s1_stride)

            x_sub = x_sub.permute(0, 2, 3, 4, 5, 1)
            x_sub = x_sub.reshape(
                patched.size(0),
                s2_patched + 2 * padding[1],
                s1_patched + 2 * padding[0],
                -1,
            )
            x_sub = x_sub.permute(0, 3, 2, 1)

            patched = torch.cat((patched, x_sub), 1)

        return patched

    def _unpad(self, x):
        """Remove padding around the edges (`mode=circular`) of tensor `x`.

        Parameters
        ----------
        x : torch.tensor
            padded input tensor, shape (b, c, h + self.padding_height * 2, w + self.padding_width * 2)

        Returns
        -------
        x : torch.tensor
            unpadded tensor, shape (b, c, h, w)
        """
        return x[
            ...,
            self.padding_height : -self.padding_height,
            self.padding_width : -self.padding_width,
        ].contiguous()


def make_patches(x, n, p=0):
    """make_patches splits `x` into `n` equally-sized patches
    with padding fraction `p`. Stacks patches along the batch dimension.

    Starting with an input tensor of shape (batch, C, s) or (batch, C, h, w),
    returns a corresponding patched output tensor of shape (n * batch, C, s / n + 2p) 
    or (n1 * n2 * batch, C, h / n1 + 2 * p1, w / n2 + 2 * p2)


    Parameters
    ----------
    x : torch.tensor
        input tensor, before patching
    n : int
        number of patches into which to split each example in `x`
    p : int, optional
        number of pixels to use when padding `x`, by default 0
    """

    size = x.size()

    # Only 1D and 2D supported
    assert len(size) == 3 or len(size) == 4

    if len(size) == 3:
        d = 1
    else:
        d = 2

    if isinstance(p, int):
        p = [p, p]

    # Pad
    if p[0] > 0 or p[1] > 0:
        if d == 1:
            x = torch.nn.functional.pad(x, pad=p, mode="circular")
        else:
            x = torch.nn.functional.pad(
                x, pad=[p[1], p[1], p[0], p[0]], mode="circular"
            )

    if isinstance(n, int):
        n = [n, n]

    if n[0] <= 1 and n[1] <= 1:
        return x

    # Patches must be equally sized
    for j in range(d):
        assert size[-(j + 1)] % n[-(j + 1)] == 0

    # Patch
    for j in range(d):
        patch_size = size[-(j + 1)] // n[-(j + 1)]
        x = x.unfold(-(2 * j + 1), patch_size + 2 * p[-(j + 1)], patch_size)

    x = x.permute(0, 2, 3, 4, 5, 1)
    x = x.reshape(
        size[0] * n[0] * n[1],
        size[-1] // n[-1] + 2 * p[-1],
        size[-2] // n[-2] + 2 * p[-2],
        size[1],
    )
    x = x.permute(0, 3, 2, 1)

    return x
