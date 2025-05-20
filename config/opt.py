from typing import Optional, Literal
from zencfg import ConfigBase

class OptimizationConfig(ConfigBase):
    """OptimizationConfig provides options for configuring training hyperparameters,
    like the number of epochs, loss, learning rate, and LR schedule. 

    Parameters
    ----------
    n_epochs: int
        Number of training epochs to run
    training_loss: Literal['h1', 'l2'] = "h1"
        Tag of training loss to use. All training losses
        should be norms in function space. You may also
        provide your own key-value pair behavior for custom losses
        in your training scripts.

        * If 'h1', uses the Sobolev norm

        * If 'l2', uses L2-norm. 
    learning_rate: float = 3e-4
        Learning rate for training
    weight_decay: float = 1e-4
        Weight decay to apply during training
    eval_interval: int = 1
        Number of training epochs between each epoch of evaluation
    mixed_precision: bool = False
        Whether to use automatic mixed precision (AMP) via ``torch.autocast``,
        e.g. for half-precision model weights. 
    scheduler: Literal['StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR'] = "StepLR"
        Name of PyTorch LR scheduler to use. We implement scripts with these options,
        but users are free to implement their own key-value behaviorfor any scheduler
        of their choice. 
    scheduler_T_max: int = 500
        Maximum training epochs. Some learning rate schedulers take `T_max`
        as a hyperparameter to control the steepness of the LR schedule.
    scheduler_patience: int = 50
        Patience hyperparameter for ``ReduceLROnPlateau``. Generally,
        the number of stagnating epochs to wait before decreasing the LR by
        a factor of ``gamma``. 
    step_size: int = 100
        step size hyperparameter for ``StepLR``. Number of steps to wait before
        decreasing LR by a factor of ``gamma``. 
    gamma: float = 0.5
        Factor by which to decrease LR in step-based LR schedulers. 
    """
    n_epochs: int
    training_loss: Literal['h1', 'l2'] = "h1"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    eval_interval: int = 1
    mixed_precision: bool = False
    scheduler: Literal['StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR'] = "StepLR"
    scheduler_T_max: int = 500
    scheduler_patience: int = 50
    step_size: int = 100
    gamma: float = 0.5

class PatchingConfig(ConfigBase):
    levels: int = 0
    padding: int = 0
    stitching: bool = False