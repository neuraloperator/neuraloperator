import torch
from neuralop.training.patching import MultigridPatching2D

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

    def preprocess(self, data_dict, batched=True):
        x = data_dict['x'].to(self.device)
        y = data_dict['y'].to(self.device)

        if self.in_normalizer is not None:
            x = self.in_normalizer.transform(x)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x, batched=batched)
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
                 device: str='cpu', in_normalizer=None, out_normalizer=None,
                 positional_encoding=None):
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
        positional_encoding : neuralop.datasets.transforms.PositionalEmbedding2D, optional
            appends pos encoding to x if used
        device : str, optional
            device 'cuda' or 'cpu' where computations are performed
        positional_encoding : neuralop.datasets.transforms.Transform, optional
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
        self.positional_encoding = positional_encoding
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
    
    def preprocess(self, data_dict, batched=True):
        """
        Preprocess data assuming that if encoder exists, it has 
        encoded all data during data loading
        
        Params
        ------

        data_dict: dict
            dictionary keyed with 'x', 'y' etc
            represents one batch of data input to a model
        batched: bool
            whether the first dimension of 'x', 'y' represents batching
        """
        data_dict = {k:v.to(self.device) for k,v in data_dict.items() if torch.is_tensor(v)}
        x,y = data_dict['x'], data_dict['y']
        if self.in_normalizer:
            x = self.in_normalizer.transform(x)
            y = self.out_normalizer.transform(y)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x, batched=batched)
        data_dict['x'],data_dict['y'] = self.patcher.patch(x,y)
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
        y = data_dict['y']
        out,y = self.patcher.unpatch(out,y)

        if self.out_normalizer:
            y = self.out_normalizer.inverse_transform(y)
            out = self.out_normalizer.inverse_transform(out)
        
        data_dict['y'] = y

        return out, data_dict

    def forward(self, **data_dict):
        data_dict = self.preprocess(data_dict)
        output = self.model(**data_dict)
        output, data_dict = self.postprocess(output, data_dict)
        return output, data_dict
