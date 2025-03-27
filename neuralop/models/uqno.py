from copy import deepcopy
from pathlib import Path

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
        self._version = '0.2.0'

        self.base_model = base_model
        if residual_model is None:
            residual_model = deepcopy(base_model)
        self.residual_model = residual_model

        self._model_cls_names = {'solution': self.base_model.__class__.__name__,
                                 'residual': self.residual_model.__class__.__name__}
    
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
    
    def save_checkpoint(self, save_folder, save_name):
        """Saves the model state and init param in the given folder under the given name
        """
        save_folder = Path(save_folder)
        if not save_folder.exists():
            save_folder.mkdir(parents=True)

        # save the solution and residual models separately to avoid serializing the individual modules

        ## save solution model
        solution_model_state_dict_filepath = save_folder.joinpath(f'{save_name}_solution_state_dict.pt').as_posix()
        torch.save(self.base_model.state_dict(), solution_model_state_dict_filepath)

        solution_metadata_filepath = save_folder.joinpath(f'{save_name}_solution_metadata.pkl').as_posix()
        torch.save(self.base_model._init_kwargs, solution_metadata_filepath)

        ## save residual model
        residual_model_state_dict_filepath = save_folder.joinpath(f'{save_name}_residual_state_dict.pt').as_posix()
        torch.save(self.residual_model.state_dict(), residual_model_state_dict_filepath)

        residual_metadata_filepath = save_folder.joinpath(f'{save_name}_residual_metadata.pkl').as_posix()
        torch.save(self.residual_model._init_kwargs, residual_metadata_filepath)

        ## save init kwargs
        metadata_filepath = save_folder.joinpath(f'{save_name}_submodule_cls_names.pkl').as_posix()
        torch.save(self._model_cls_names, metadata_filepath)
    
    def load_checkpoint(self, save_folder, save_name, map_location=None):
        save_folder = Path(save_folder)

        self.base_model.load_checkpoint(save_folder, f"{save_name}_solution", map_location=map_location)   
        self.residual_model.load_checkpoint(save_folder, f"{save_name}_residual", map_location=map_location)   

    @classmethod
    def from_checkpoint(self, save_folder, save_name, map_location=None):
        save_folder = Path(save_folder)

        # load the model class names to initialize separately
        cls_names_filepath = save_folder.joinpath(f'{save_name}_submodule_cls_names.pkl').as_posix()
        cls_names = torch.load(cls_names_filepath)

        # load the solution and residual models separately to avoid ``weights_only=False`` loading
        solution_model_cls = BaseModel._models[cls_names['solution']]
        solution_model = solution_model_cls.from_checkpoint(save_folder, save_name=f"{save_name}_solution", map_location=map_location)

        residual_model_cls = BaseModel._models[cls_names['residual']]
        residual_model = residual_model_cls.from_checkpoint(save_folder, save_name=f"{save_name}_residual", map_location=map_location)

        return UQNO(base_model=solution_model, residual_model=residual_model)
