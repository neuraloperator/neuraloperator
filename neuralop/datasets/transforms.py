from abc import abstractmethod
import torch
from .positional_encoding import append_2d_grid_positional_encoding
from torch.utils.data import Dataset

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


class PositionalEmbedding():
    def __init__(self, grid_boundaries, channel_dim):
        self.grid_boundaries = grid_boundaries
        self.channel_dim = channel_dim
        self._grid = None    

    def __call__(self, data):
        return append_2d_grid_positional_encoding(data,
                                                  self.grid_boundaries,
                                                  self.channel_dim)


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
