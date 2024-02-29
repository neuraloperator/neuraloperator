import torch
import torch.nn as nn

from .base_model import BaseModel

class UQNO(BaseModel, name="UQNO"):
    """General N-dim (alpha, delta) Risk-Controlling Neural Operator
    Source: https://arxiv.org/abs/2402.01960

    The UQNO must be trained on 

    Parameters
    ----------
    model : BaseModel
        base model to generate point predictions
    alpha : float
        fraction of points excluded from codomain coverage,
        i.e. target codomain coverage rate is 1-alpha
    delta : float
        1 - delta controls the expected proportion of functions 
        that predict an overall coverage of 1-alpha within a given band
    residual_model_config : dict, optional
        config for the residual model
            if None, then the residual model will be a copy 
            of the base model
            otherwise the residual model will be initialized from the config
    """
    def __init__(self,
                 base_model: BaseModel,
                 alpha: float,
                 delta: float,
                 residual_model_config: dict = None
                 ):
        super().__init__()

        self.alpha = alpha
        self.delta = delta
        self.base_model = base_model
        if residual_model_config is not None:
