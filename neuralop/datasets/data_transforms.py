import torch
from neuralop.training.patching import MultigridPatching2D


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
        self.grid_boundaries = grid_boundaries
        self._grid = None

    def grid(self, spatial_dims, device, dtype):
        if self._grid is None:
            grid_x, grid_y = regular_grid(spatial_dims,
                                      grid_boundaries=self.grid_boundaries)
            grid_x = grid_x.to(device).to(dtype).unsqueeze(0).unsqueeze(0)
            grid_y = grid_y.to(device).to(dtype).unsqueeze(0).unsqueeze(0)
            self._grid = grid_x, grid_y

        return self._grid

    def __call__(self, data):
        batch_size = data.shape[0]
        x, y = self.grid(data.shape[-2:], data.device, data.dtype)

        return torch.cat((data, x.expand(batch_size, -1, -1, -1),
                          y.expand(batch_size, -1, -1, -1)),
                         dim=1)


class DefaultDataProcessor(torch.nn.Module):
    def __init__(self, 
                 in_normalizer=None, out_normalizer=None, 
                 positional_encoding=None):
        """A simple processor to pre/post process data before training/inferencing a model

        Parameters
        ----------
        in_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the input samples
        out_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the target and predicted samples
        positional_encoding : Processor, optional, default is None
            class that appends a positional encoding to the input
        """
        super().__init__()
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer
        self.positional_encoding = positional_encoding
        self.device = 'cpu'
    
    def wrap(self, model):
        self.model = model
        return self

    def to(self, device):
        if self.in_normalizer is not None:
            self.in_normalizer = self.in_normalizer.to(device)
        if self.out_normalizer is not None:
            self.out_normalizer = self.out_normalizer.to(device)
        self.device = device
        return self

    def preprocess(self, data_dict):
        x = data_dict['x'].to(self.device)
        y = data_dict['y'].to(self.device)

        if self.in_normalizer is not None:
            x = self.in_normalizer.transform(x)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        if self.out_normalizer is not None and self.train:
            y = self.out_normalizer.transform(y)

        data_dict['x'] = x
        data_dict['y'] = y

        return data_dict

    def postprocess(self, output, data_dict):
        y = data_dict['y']
        if self.out_normalizer and not self.train:
            output = self.out_normalizer.inverse_transform(output)
            y = self.out_normalizer.inverse_transform(y)
        data_dict['y'] = y
        return output, data_dict
    
    def forward(self, **data_dict):
        data_dict = self.preprocess(data_dict)
        output = self.model(data_dict['x'])
        output = self.postprocess(output)
        return output, data_dict

class MGPatchingDataProcessor(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, levels: int, 
                 padding_fraction: float, stitching: float, 
                 device: str='cpu', in_normalizer=None, out_normalizer=None):
        """MGPatchingDataProcessor
        Applies multigrid patching to inputs out-of-place 
        with an optional output encoder/other data transform

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
        in_normalizer : neuralop.datasets.transforms.Transform, optional
            OutputEncoder to decode model inputs, by default None
        in_normalizer : neuralop.datasets.transforms.Transform, optional
            OutputEncoder to decode model outputs, by default None
        device : str, optional
            device 'cuda' or 'cpu' where computations are performed
        """
        super().__init__()
        self.levels = levels
        self.padding_fraction = padding_fraction
        self.stitching = stitching
        self.patcher = MultigridPatching2D(model=model, levels=self.levels, 
                                      padding_fraction=self.padding_fraction,
                                      stitching=self.stitching)
        self.device = device
        
        # set normalizers to none by default
        self.in_normalizer, self.out_normalizer = None, None
        if in_normalizer:
            self.in_normalizer = in_normalizer.to(self.device)
        if out_normalizer:
            self.out_normalizer = out_normalizer.to(self.device)
        self.model = None
    
    def to(self, device):
        self.device = device
        if self.in_normalizer:
            self.in_normalizer = self.in_normalizer.to(self.device)
        if self.out_normalizer:
            self.out_normalizer = self.out_normalizer.to(self.device)
    
    def wrap(self, model):
        self.model = model
        return self
    
    def preprocess(self, data_dict):
        """
        Preprocess data assuming that if encoder exists, it has 
        encoded all data during data loading
        
        Params
        ------

        data_dict: dict
            dictionary keyed with 'x', 'y' etc
            represents one batch of data input to a model
        """
        data_dict = {k:v.to(self.device) for k,v in data_dict.items()}
        if self.in_normalizer:
            data_dict['x'] = self.in_normalizer.transform(data_dict)
            data_dict['y'] = self.out_normalizer.transform(data_dict)
        data_dict['x'],data_dict['y'] = self.patcher.patch(data_dict['x'],data_dict['y'])
        return data_dict
    
    def postprocess(self, out, data_dict):
        """
        Postprocess model outputs, including decoding
        if an encoder exists.
        
        Params
        ------

        data_dict: dict
            dictionary keyed with 'x', 'y' etc
            represents one batch of data input to a model
        out: torch.Tensor 
            model output predictions
        """

        x,y = self.patcher.unpatch(data_dict['x'],data_dict['y'])

        if self.out_normalizer:
            y = self.out_normalizer.inverse_transform(y)
            out = self.out_normalizer.inverse_transform(out)
        
        data_dict['x'] = x
        data_dict['y'] = y

        return out, data_dict

    def forward(self, **data_dict):
        data_dict = self.preprocess(data_dict)
        output = self.model(**data_dict)
        output, data_dict = self.postprocess(output, data_dict)
        return output, data_dict