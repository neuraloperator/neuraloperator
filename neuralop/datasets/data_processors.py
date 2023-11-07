"""
DataProcessors handle transformations of data for downstream use
in training and testing Neural Operator models.
"""

from abc import ABCMeta, abstractmethod
from typing import List

import torch
from torch import nn

from neuralop.training.patching import MultigridPatching2D
from .transforms import Transform

class DataProcessor(ABCMeta):
    """
    Abstract base class for DataProcessor objects
    """
    def __init__(self):
        pass

    @abstractmethod
    def preprocess(self, sample):
        pass

    @abstractmethod
    def postprocess(self, sample):
        pass

class MGPatchingDataProcessor(DataProcessor):
    def __init__(self, model: nn.Module, levels: int, padding_fraction: float, 
                 stitching: float, device: str='cpu', transforms=None):
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
        encoder : neuralop.datasets.transforms.Transform, optional
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
        self.transforms = transforms.to(self.device)
        
        ## TODO @Jean: where should loading to device occur within the DataProcessor?
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
        if self.transforms:
            sample = self.transforms.transform(sample)
        sample['x'] = self.patcher.patch(sample['x'])
        sample['y'] = self.patcher.patch(sample['y'])
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

        y = self.patcher.unpatch(sample['y'])
        out = self.patcher.unpatch(out)

        if self.transforms:
            y = self.transforms.inverse_transform(y)
            out = self.transforms.inverse_transform(out)
        
        sample['y'] = y

        return out, sample

class Composite(DataProcessor):
    def __init__(self, transforms: List[Transform], device: str='cpu'):
        """Composite DataProcessor applies a sequence of transforms
        to input data, and applies their inverses in reverse to model outputs

        Parameters
        ----------
        transforms : List[Transform]
            list of Transform objects to apply to each sample, in order
        device : str, optional
            device on which to store data, by default 'cpu'
        """
        super().__init__()
        assert type(transforms) == list, \
            "Error: Composite transform must be initialized with a list"
        self.device = device
        self.transforms = transforms.to(self.device)
    
    def preprocess(self, sample: dict):
        """preprocess a sample, assuming data dict format

        Parameters
        ----------
        sample : dict
            data with at least 'x' and 'y' keys
        """
        sample = {k: v.to(self.device) for k,v in sample.items()}
        for tform in self.transforms:
            sample = tform.transform(sample)
        return sample
    
    def postprocess(self, sample: dict, out: torch.Tensor):
        """preprocess a sample, assuming data dict format

        Parameters
        ----------
        sample : dict
            data with at least 'x' and 'y' keys
        out : torch.tensor
            predictions output by model
        """
        for tform in self.transforms[::-1]:
            sample = tform.inverse_transform(sample)
        return sample
        

        