from abc import ABCMeta, abstractmethod

import torch
from neuralop.training.patching import MultigridPatching2D


class DataProcessor(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self):
        """DataProcessor exposes functionality for pre-
        and post-processing data during training or inference.

        To be a valid DataProcessor within the Trainer requires
        that the following methods are implemented:

        - to(device): load necessary information to device, in keeping
            with PyTorch convention
        - preprocess(data): processes data from a new batch before being
            put through a model's forward pass
        - postprocess(out): processes the outputs of a model's forward pass
            before loss and backward pass
        - wrap(self, model):
            wraps a model in preprocess and postprocess steps to create one forward pass
        - forward(self, x):
            forward pass providing that a model has been wrapped
        """
        super().__init__()

    @abstractmethod
    def to(self, device):
        pass

    @abstractmethod
    def preprocess(self, x):
        pass

    @abstractmethod
    def postprocess(self, x):
        pass

    @abstractmethod
    def wrap(self, model):
        pass

    @abstractmethod
    def forward(self, x):
        pass


class DefaultDataProcessor(DataProcessor):
    def __init__(
        self, in_normalizer=None, out_normalizer=None, positional_encoding=None
    ):
        """A simple processor to pre/post process data before training/inferencing a model.

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
        self.device = "cpu"

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
        x = data_dict["x"].to(self.device)
        y = data_dict["y"].to(self.device)

        if self.in_normalizer is not None:
            x = self.in_normalizer.transform(x)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x, batched=batched)
        if self.out_normalizer is not None and self.train:
            y = self.out_normalizer.transform(y)

        data_dict["x"] = x
        data_dict["y"] = y

        return data_dict

    def postprocess(self, output, data_dict):
        y = data_dict["y"]
        if self.out_normalizer and not self.train:
            output = self.out_normalizer.inverse_transform(output)
            y = self.out_normalizer.inverse_transform(y)
        data_dict["y"] = y
        return output, data_dict

    def forward(self, **data_dict):
        data_dict = self.preprocess(data_dict)
        output = self.model(data_dict["x"])
        output = self.postprocess(output)
        return output, data_dict

class IncrementalDataProcessor(torch.nn.Module):
    def __init__(self, 
                 in_normalizer=None, out_normalizer=None, 
                 positional_encoding=None, device = 'cpu',
                 subsampling_rates=[2, 1], dataset_resolution=16, dataset_indices=[2,3], epoch_gap=10, verbose=False):
        """An incremental processor to pre/post process data before training/inferencing a model
        In particular this processor first regularizes the input resolution based on the sub_list and dataset_indices
        in the spatial domain based on a fixed number of epochs. We incrementally increase the resolution like done 
        in curriculum learning to train the model. This is useful for training models on large datasets with high
        resolution data.

        Parameters
        ----------
        in_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the input samples
        out_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the target and predicted samples
        positional_encoding : Processor, optional, default is None
            class that appends a positional encoding to the input
        device : str, optional, default is 'cpu'
            device 'cuda' or 'cpu' where computations are performed
        subsampling_rates : list, optional, default is [2, 1]
            list of subsampling rates to use
        dataset_resolution : int, optional, default is 16
            resolution of the input data
        dataset_indices : list, optional, default is [2, 3]
            list of indices of the dataset to slice to regularize the input resolution - Spatial Dimensions
        epoch_gap : int, optional, default is 10
            number of epochs to wait before increasing the resolution
        verbose : bool, optional, default is False
            if True, print the current resolution
        """
        super().__init__()
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer
        self.positional_encoding = positional_encoding
        self.device = device
        self.sub_list = subsampling_rates
        self.dataset_resolution = dataset_resolution
        self.dataset_indices = dataset_indices
        self.epoch_gap = epoch_gap
        self.verbose = verbose
        self.mode = "Train"
        self.epoch = 0
        
        self.current_index = 0
        self.current_logged_epoch = 0
        self.current_sub = self.index_to_sub_from_table(self.current_index)
        self.current_res = int(self.dataset_resolution / self.current_sub)   
        
        print(f'Original Incre Res: change index to {self.current_index}')
        print(f'Original Incre Res: change sub to {self.current_sub}')
        print(f'Original Incre Res: change res to {self.current_res}')
            
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
    
    def epoch_wise_res_increase(self, epoch):
        # Update the current_sub and current_res values based on the epoch
        if epoch % self.epoch_gap == 0 and epoch != 0 and (
                self.current_logged_epoch != epoch):
            self.current_index += 1
            self.current_sub = self.index_to_sub_from_table(self.current_index)
            self.current_res = int(self.dataset_resolution / self.current_sub)
            self.current_logged_epoch = epoch

            if self.verbose:
                print(f'Incre Res Update: change index to {self.current_index}')
                print(f'Incre Res Update: change sub to {self.current_sub}')
                print(f'Incre Res Update: change res to {self.current_res}')

    def index_to_sub_from_table(self, index):
        # Get the sub value from the sub_list based on the index
        if index >= len(self.sub_list):
            return self.sub_list[-1]
        else:
            return self.sub_list[index]

    def regularize_input_res(self, x, y):
        # Regularize the input data based on the current_sub and dataset_name
        for idx in self.dataset_indices:
            indexes = torch.arange(0, x.size(idx), self.current_sub, device=self.device)
            x = x.index_select(dim=idx, index=indexes)
            y = y.index_select(dim=idx, index=indexes)
        return x, y
    
    def step(self, loss=None, epoch=None, x=None, y=None):
        if x is not None and y is not None:
            self.epoch_wise_res_increase(epoch)
            return self.regularize_input_res(x, y)
        
    def preprocess(self, data_dict, batched=True):
        x = data_dict['x'].to(self.device)
        y = data_dict['y'].to(self.device)

        if self.in_normalizer is not None:
            x = self.in_normalizer.transform(x)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x, batched=batched)
        if self.out_normalizer is not None and self.train:
            y = self.out_normalizer.transform(y)
        
        if self.mode == "Train":
            x, y = self.step(epoch=self.epoch, x=x, y=y)
        
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
    
class MGPatchingDataProcessor(DataProcessor):
    def __init__(
        self,
        model: torch.nn.Module,
        levels: int,
        padding_fraction: float,
        stitching: float,
        device: str = "cpu",
        in_normalizer=None,
        out_normalizer=None,
        positional_encoding=None,
    ):
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
        self.patcher = MultigridPatching2D(
            model=model,
            levels=self.levels,
            padding_fraction=self.padding_fraction,
            stitching=self.stitching,
        )
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
        data_dict = {
            k: v.to(self.device) for k, v in data_dict.items() if torch.is_tensor(v)
        }
        x, y = data_dict["x"], data_dict["y"]
        if self.in_normalizer:
            x = self.in_normalizer.transform(x)
            y = self.out_normalizer.transform(y)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x, batched=batched)
        data_dict["x"], data_dict["y"] = self.patcher.patch(x, y)
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
        y = data_dict["y"]
        out, y = self.patcher.unpatch(out, y)

        if self.out_normalizer:
            y = self.out_normalizer.inverse_transform(y)
            out = self.out_normalizer.inverse_transform(out)

        data_dict["y"] = y

        return out, data_dict

    def forward(self, **data_dict):
        data_dict = self.preprocess(data_dict)
        output = self.model(**data_dict)
        output, data_dict = self.postprocess(output, data_dict)
        return output, data_dict
