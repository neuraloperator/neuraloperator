import torch


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

    def postprocess(self, output):
        if self.out_normalizer and not self.train:
            output = self.out_normalizer.inverse_transform(output)
        return output
    
    def forward(self, **data_dict):
        data_dict = self.preprocess(data_dict)
        output = self.model(data_dict['x'])
        output = self.postprocess(output)
        return output, data_dict
