"""
Base Loss class for neuralop
"""
from  abc import ABCMeta, abstractmethod
from torch import nn

class Loss(nn.Module, ABCMeta):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self):
        # all losses must implement forward
        pass

    @property
    def name(self):
        '''
        Default name for logging, override for more specificity
        '''
        return self.__class__.__name__