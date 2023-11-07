"""
DataPipelines handle transformations of data for downstream use
in training and testing Neural Operator models.
"""

from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn

from neuralop.training.patching import MultigridPatching2D
from .transforms import Transform

class DataPipeline(ABC):
    """
    Abstract base class for DataPipeline objects
    """
    def __init__(self):
        pass

    @abstractmethod
    def preprocess(self, sample):
        pass

    @abstractmethod
    def postprocess(self, sample):
        pass

class MGPatchingDataPipeline(DataPipeline):
    def __init__(self, model: nn.Module, levels: int, padding_fraction: float, 
                 stitching: float, device: str='cpu', in_normalizer=None, out_normalizer=None):
        """MGPatchingDataPipeline 
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
        device 
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
        
        ## TODO @Jean: where should loading to device occur within the DataPipeline?
        #   should it be a default behavior inside the trainer
        #   or should users manage this themselves?
    
    def preprocess(self, sample):
        """
        Preprocess data assuming that if encoder exists, it has 
        encoded all data during data loading
        
        Params
        ------

        sample: dict
            dictionary keyed with 'x', 'y' etc
            represents one batch of data input to a model
        """
        sample = {k:v.to(self.device) for k,v in sample.items()}
        if self.in_normalizer:
            sample = self.in_normalizer.transform(sample)
        sample['x'],sample['y'] = self.patcher.patch(sample['x'],sample['y'])
        return sample
    
    def postprocess(self, out, sample):
        """
        Postprocess model outputs, including decoding
        if an encoder exists.
        
        Params
        ------

        sample: dict
            dictionary keyed with 'x', 'y' etc
            represents one batch of data input to a model
        out: torch.Tensor 
            model output predictions
        """

        x,y = self.patcher.unpatch(sample['x'],sample['y'])

        if self.out_normalizer:
            y = self.out_normalizer.inverse_transform(y)
            out = self.out_normalizer.inverse_transform(out)
        
        sample['x'] = x
        sample['y'] = y

        return out, sample

class Composite(DataPipeline):
    def __init__(self, in_transforms: List[Transform], out_transforms: List[Transform]=None, device: str='cpu'):
        """Composite DataPipeline applies a sequence of transforms
        to input data, and applies their inverses in reverse to model outputs

        Parameters
        ----------
        in_transforms : List[Transform]
            list of Transform objects to apply to each sample's input data, in order
        out_transforms : List[Transform], optional
            list of Transform objects to apply to each sample's truth and model output data, in order
            If none is provided, then in_transforms are applied to everything
        device : str, optional
            device on which to store data, by default 'cpu'
        """
        super().__init__()
        assert type(in_transforms) == list, \
            "Error: Composite transform must be initialized with a list of in_transforms"
        assert type(out_transforms) == list, \
            "Error: Composite transform must be initialized with a list of out_transforms"
        self.device = device
        self.in_transforms = in_transforms.to(self.device)
        if self.out_transforms:
            self.out_transforms = out_transforms.to(self.device)
        else:
            self.out_transforms = None
    
    def preprocess(self, sample: dict):
        """preprocess a sample, assuming data dict format

        Parameters
        ----------
        sample : dict
            data with at least 'x' and 'y' keys
        """
        sample = {k: v.to(self.device) for k,v in sample.items()}
        # apply input transformation to 'x'
        x = sample['x']
        y = sample['y']
        for tform in self.in_transforms:
            x = tform.transform(x)
        
        if self.out_transforms:
            for tform in self.out_transforms:
                y = tform.transform(y)
        else:
            for tform in self.in_transforms:
                y = tform.transform(y)
        sample['x'] = x
        sample['y'] = y

        del x,y
        return sample
    
    def postprocess(self, out: torch.Tensor, sample: dict):
        """preprocess a sample, assuming data dict format

        Parameters
        ----------
        sample : dict
            data with at least 'x' and 'y' keys
        out : torch.tensor
            predictions output by model
        """
        y = sample['y']
        if self.out_transforms:
            for tform in self.out_transforms[::-1]:
                y = tform.inverse_transform(y)
                out = tform.inverse_transform(out)
        else:
            for tform in self.in_transforms[::-1]:
                y = tform.inverse_transform(y)
                out = tform.inverse_transform(out)

        sample['y'] = y
        return out, sample
    
        

        