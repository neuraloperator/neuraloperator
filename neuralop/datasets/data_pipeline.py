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

class DNEPipeline(DataPipeline):
    def __init__(self, 
                 in_normalizer: Transform,
                 out_normalizer: Transform,
                 pos_encoding: nn.Module=None,
                 device: str='cpu'):
        """DNE DataPipeline sends data to Device, 
        applies a Normalizer to input data, 
        then applies optional positional 
        Encoding.

        Parameters
        ----------
        in_normalizer : List[Transform], optional
            normalizer to apply to each sample's input data, in order
        out_normalizer : List[Transform], optional
            normalizer to apply to each sample's truth and 
            model output data, in order
        pos_encoding: nn.Module, optional
            optional layer to append positional 
            encoding onto input data
        device : str, optional
            device on which to store data, by default 'cpu'
        """
        super().__init__()
        self.device = device

        if in_normalizer:
            self.in_normalizer = in_normalizer.to(self.device)
        else:
            self.in_normalizer = None

        if out_normalizer:
            self.out_normalizer = out_normalizer.to(self.device)
        else:
            self.out_normalizer = None
        
        self.pos_encoding = pos_encoding
        

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
        if self.in_normalizer:
            x = self.in_normalizer.transform(x)
        
        if self.out_normalizer:
            y = self.out_normalizer.transform(y)
        
        if self.pos_encoding:
            x = self.pos_encoding(x)
        sample['x'] = x
        sample['y'] = y

        del x,y
        return sample
    
    def postprocess(self, out: torch.Tensor, sample: dict):
        """postprocess a sample, assuming data dict format

        Parameters
        ----------
        out : torch.Tensor
            predictions output by model
        sample : dict
            data with at least 'x' and 'y' keys
        """
        y = sample['y']
        if self.out_normalizer:
            y = self.out_normalizer.inverse_transform(y)
            out = self.out_normalizer.inverse_transform(out)

        sample['y'] = y
        return out, sample
    
        

        