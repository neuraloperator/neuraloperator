from abc import abstractmethod
from typing import List

import torch
from torch.utils.data import Dataset
from neuralop.training.patching import MultigridPatching2D

class Transform(torch.nn.Module):
    """
    Applies transforms or inverse transforms to 
    model inputs or outputs, respectively
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def inverse_transform(self):
        pass

    @abstractmethod
    def cuda(self):
        pass

    @abstractmethod
    def cpu(self):
        pass

    @abstractmethod
    def to(self, device):
        pass
    
class Normalizer():
    def __init__(self, mean, std, eps=1e-6):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, data):
        return (data - self.mean)/(self.std + self.eps)

class Composite(Transform):
    def __init__(self, transforms: List[Transform]):
        """Composite transform composes a list of
        Transforms into one Transform object.

        Transformations are not assumed to be commutative

        Parameters
        ----------
        transforms : List[Transform]
            list of transforms to be applied to data
            in order
        """
        super.__init__()
        self.transforms = transforms
    
    def transform(self, data_dict):
        for tform in self.transforms:
            data_dict = tform.transform(self.data_dict)
        return data_dict
    
    def inverse_transform(self, data_dict):
        for tform in self.transforms[::-1]:
            data_dict = tform.transform(self.data_dict)
        return data_dict

    def to(self, device):
        # all Transforms are required to implement .to()
        self.transforms = [t.to(device) for t in self.transforms if hasattr(t, 'to')]
        return self

class MGPatchingTransform(Transform):
    def __init__(self, model: torch.nn.Module, levels: int, 
                 padding_fraction: float, stitching: float):
        """Wraps MultigridPatching2D to expose canonical
        transform .transform() and .inverse_transform() API

        Parameters
        ----------
        model: nn.Module
            model to wrap in MultigridPatching2D
        levels : int
            mg_patching level parameter for MultigridPatching2D
        padding_fraction : float
            mg_padding_fraction parameter for MultigridPatching2D
        stitching : float
            mg_patching_stitching parameter for MultigridPatching2D
        """
        super.__init__()

        self.levels = levels
        self.padding_fraction = padding_fraction
        self.stitching = stitching
        self.patcher = MultigridPatching2D(model=model, levels=self.levels, 
                                      padding_fraction=self.padding_fraction,
                                      stitching=self.stitching)
    def transform(self, data_dict):
        
        x = data_dict['x']
        y = data_dict['y']

        x,y = self.patcher.patch(x,y)

        data_dict['x'] = x
        data_dict['y'] = y
        return data_dict
    
    def inverse_transform(self, data_dict):
        x = data_dict['x']
        y = data_dict['y']

        x,y = self.patcher.unpatch(x,y)

        data_dict['x'] = x
        data_dict['y'] = y
        return data_dict
    
    def to(self, _):
        # nothing to pass to device
        return self

class RandomMGPatch():
    def __init__(self, levels=2):
        self.levels = levels
        self.step = 2**levels

    def __call__(self, data):

        def _get_patches(shifted_image, step, height, width):
            """Take as input an image and return multi-grid patches centered around the middle of the image
            """
            if step == 1:
                return (shifted_image, )
            else:
                # Notice that we need to stat cropping at start_h = (height - patch_size)//2
                # (//2 as we pad both sides)
                # Here, the extracted patch-size is half the size so patch-size = height//2
                # Hence the values height//4 and width // 4
                start_h = height//4
                start_w = width//4

                patches = _get_patches(shifted_image[:, start_h:-start_h, start_w:-start_w], step//2, height//2, width//2)

                return (shifted_image[:, ::step, ::step], *patches)
        
        x, y = data
        channels, height, width = x.shape
        center_h = height//2
        center_w = width//2

        # Sample a random patching position
        pos_h = torch.randint(low=0, high=height, size=(1,))[0]
        pos_w = torch.randint(low=0, high=width, size=(1,))[0]

        shift_h = center_h - pos_h
        shift_w = center_w - pos_w

        shifted_x = torch.roll(x, (shift_h, shift_w), dims=(0, 1))
        patches_x = _get_patches(shifted_x, self.step, height, width)
        shifted_y = torch.roll(y, (shift_h, shift_w), dims=(0, 1))
        patches_y = _get_patches(shifted_y, self.step, height, width)

        return torch.cat(patches_x, dim=0), patches_y[-1]

class MGPTensorDataset(Dataset):
    def __init__(self, x, y, levels=2):
        assert (x.size(0) == y.size(0)), "Size mismatch between tensors"
        self.x = x
        self.y = y
        self.levels = 2
        self.transform = RandomMGPatch(levels=levels)

    def __getitem__(self, index):
        return self.transform((self.x[index], self.y[index]))

    def __len__(self):
        return self.x.size(0)
    

def regular_grid(spatial_dims, grid_boundaries=[[0, 1], [0, 1]]):
    """
    Appends grid positional encoding to an input tensor, concatenating as additional dimensions along the channels
    """
    height, width = spatial_dims

    xt = torch.linspace(grid_boundaries[0][0], grid_boundaries[0][1],
                        height + 1)[:-1]
    yt = torch.linspace(grid_boundaries[1][0], grid_boundaries[1][1],
                        width + 1)[:-1]

    grid_x, grid_y = torch.meshgrid(xt, yt, indexing='ij')

    grid_x = grid_x.repeat(1, 1)
    grid_y = grid_y.repeat(1, 1)

    return grid_x, grid_y


class PositionalEmbedding2D():
    """A simple positional embedding as a regular 2D grid
    """
    def __init__(self, grid_boundaries=[[0, 1], [0, 1]]):
        """PositionalEmbedding2D applies a simple positional 
        embedding as a regular 2D grid

        Parameters
        ----------
        grid_boundaries : list, optional
            coordinate boundaries of input grid, by default [[0, 1], [0, 1]]
        """
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None

    def grid(self, spatial_dims, device, dtype):
        """grid generates 2D grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : torch.size
             sizes of spatial resolution
        device : literal 'cpu' or 'cuda:*'
            where to load data
        dtype : str
            dtype to encode data

        Returns
        -------
        torch.tensor
            output grids to concatenate 
        """
        # handle case of multiple train resolutions
        if self._grid is None or self._res != spatial_dims: 
            grid_x, grid_y = regular_grid(spatial_dims,
                                      grid_boundaries=self.grid_boundaries)
            grid_x = grid_x.to(device).to(dtype).unsqueeze(0).unsqueeze(0)
            grid_y = grid_y.to(device).to(dtype).unsqueeze(0).unsqueeze(0)
            self._grid = grid_x, grid_y
            self._res = spatial_dims

        return self._grid

    def __call__(self, data, batched=True):
        if not batched:
            if data.ndim == 3:
                data = data.unsqueeze(0)
        batch_size = data.shape[0]
        x, y = self.grid(data.shape[-2:], data.device, data.dtype)
        out =  torch.cat((data, x.expand(batch_size, -1, -1, -1),
                          y.expand(batch_size, -1, -1, -1)),
                         dim=1)
        # in the unbatched case, the dataloader will stack N 
        # examples with no batch dim to create one
        if not batched and batch_size == 1: 
            return out.squeeze(0)
        else:
            return out