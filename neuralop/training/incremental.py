""" 
Trainer for Incremental-FNO
"""

from .trainer import Trainer

class IncrementalTrainer(Trainer):
    """IncrementalTrainer subclasses the Trainer 
    to implement specific logic for the Incremental-FNO
    as described in [1]
    """
        
    def __init__(self,
                incremental_grad: bool = False, 
                incremental_loss_gap: bool = False, 
                incremental_grad_eps: float = 0.001,
                incremental_buffer: int = 5, 
                incremental_max_iter: int = 1, 
                incremental_grad_max_iter: int = 10,
                incremental_loss_eps: float = 0.001
                ):
        
        
        super().__init__()
        self.incremental_loss_gap = incremental_loss_gap
        self.incremental_grad = incremental_grad
        self.incremental = self.incremental_loss_gap or self.incremental_grad
        assert self.incremental, \
            "Error: IncrementalTrainer expects at least one incremental algorithm to be True."
        assert not (self.incremental_loss_gap and self.incremental_grad),\
            "Error: IncrementalTrainer expects only one incremental algorithm to be True."
        
        self.incremental_grad_eps = incremental_grad_eps
        self.incremental_buffer = incremental_buffer
        self.incremental_max_iter = incremental_max_iter
        self.incremental_grad_max_iter = incremental_grad_max_iter
        self.incremental_loss_eps = incremental_loss_eps
        self.loss_list = []

"""
References:

[1] George, R., Zhao, J., Kossaifi, J., Li, Z., and Anandkumar, A. (2024)
"Incremental Spatial and Spectral Learning of Neural Operators for Solving Large-Scale PDEs".
ArXiv preprint, https://arxiv.org/pdf/2211.15188
"""