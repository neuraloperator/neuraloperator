"""
Preprocessors handle transformations of data for downstream use
in training and testing Neural Operator models.
"""

from abc import ABCMeta, abstractmethod
from torch import nn

from neuralop.training.patching import MultigridPatching2D

class Preprocessor(ABCMeta):
    """
    Abstract base class for preprocessor objects
    """
    def __init__(self):
        pass

    @abstractmethod
    def preprocess(self, sample):
        pass

    @abstractmethod
    def postprocess(self, sample):
        pass

class MGPatchingPreprocessor(Preprocessor):
    def __init__(self, model: nn.Module, levels: int, padding_fraction: float, 
                 stitching: float, encoder=None):
        """MGPatchingPreprocessor 
        Applies multigrid patching to inputs out-of-place 
        with an optional output encoder

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
        encoder : neuralop.datasets.output_encoder.OutputEncoder, optional
            OutputEncoder to decode model outputs, by default None
        """
        super().__init__()
        self.levels = levels
        self.padding_fraction = padding_fraction
        self.stitching = stitching
        self.encoder = encoder
        self.patcher = MultigridPatching2D(model=model, levels=self.levels, 
                                      padding_fraction=self.padding_fraction,
                                      stitching=self.stitching)
        
        ## TODO @Jean: where should loading to device occur within the preprocessor?
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

        if self.encoder:
            y = self.encoder.decode(y)
            out = self.encoder.decode(out)
        
        sample['y'] = y

        return out, sample
        

        