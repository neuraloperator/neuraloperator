from copy import deepcopy

import torch
import torch.nn as nn


from .base_model import BaseModel

class UQNO(BaseModel, name="UQNO"):
    """General N-dim (alpha, delta) Risk-Controlling 
    Neural Operator, as described in [1]_. 

    The UQNO is trained to map input functions to a residual function
    E(a, x) that describes the predicted error between the ground truth 
    and the outputs of a trained model. E(a, x) is then used in combination
    with a calibrated scaling factor to predict calibrated uncertainty bands
    around the predictions of the trained model. 

    Parameters
    ----------
    base_model : nn.Module
        pre-trained solution operator
    residual_model : nn.Module, optional
        architecture to train as the UQNO's 
        quantile model
    
    References
    -----------
    .. [1] :

    Ma, Z., Pitt, D., Azizzadenesheli, K., and Anandkumar, A. (2024). 
        "Calibrated Uncertainty Quantification for Operator Learning
        via Conformal Prediction". TMLR, https://openreview.net/pdf?id=cGpegxy12T. 
    """
    def __init__(self,
                 base_model: nn.Module,
                 residual_model: nn.Module=None,
                 **kwargs
                 ):
        super().__init__()

        self.base_model = base_model
        if residual_model is None:
            residual_model = deepcopy(base_model)
        self.residual_model = residual_model
    
    def forward(self, *args, **kwargs):
        """
        Forward pass returns the solution u(a,x)
        and the uncertainty ball E(a,x) as a pair
        for pointwise quantile loss
        """
        self.base_model.eval() # base-model weights are frozen
        # another way to handle this would be to use LoRA, or similar
        # ie freeze the  weights, and train a low-rank matrix of weight perturbations
        with torch.no_grad():
            solution = self.base_model(*args, **kwargs)
        quantile = self.residual_model(*args, **kwargs)
        return (solution, quantile)
    