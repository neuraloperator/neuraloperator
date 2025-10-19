import torch
from torch import nn

from .trainer import Trainer
from ..models import FNO, TFNO
from ..utils import compute_explained_variance


class IncrementalFNOTrainer(Trainer):
    """
    Trainer for the Incremental Fourier Neural Operator (iFNO)

    Implements iFNO approach from [1] that progressively increases
    Fourier modes during training. This class supports two algorithms:

    1. Loss Gap (`incremental_loss_gap=True`): Increases modes when loss
       improvement becomes too small
    2. Gradient-based (`incremental_grad=True`): Uses explained variance of
       gradient strengths to determine when more modes are needed

    Parameters
    ----------
    model : nn.Module
        FNO or TFNO model to train.
    n_epochs : int
        Total number of training epochs.
    incremental_grad : bool, optional
        Use gradient-based algorithm, by default False.
    incremental_loss_gap : bool, optional
        Use loss gap algorithm, by default False.
    incremental_grad_eps : float, optional
        Explained variance threshold for gradient algorithm, by default 0.001.
    incremental_loss_eps : float, optional
        Loss improvement threshold for loss gap algorithm, by default 0.001.
    incremental_grad_max_iter : int, optional
        Iterations for gradient accumulation, by default 10.
    incremental_buffer : int, optional
        Buffer size for gradient accumulation, by default 5.

    Notes
    -----
    - Exactly one algorithm must be enabled (not both)
    - Gradient algorithm requires multiple iterations for statistics
    - Both algorithms respect maximum modes in FNO model

    References
    ----------
    .. [1] George, R., Zhao, J., Kossaifi, J., Li, Z., and Anandkumar, A. (2024)
           "Incremental Spatial and Spectral Learning of Neural Operators for Solving Large-Scale PDEs".
           TMLR, https://openreview.net/pdf?id=xI6cPQObp0.
    """

    def __init__(
        self,
        model: nn.Module,
        n_epochs: int,
        wandb_log: bool = False,
        device: str = "cpu",
        mixed_precision: bool = False,
        data_processor: nn.Module = None,
        eval_interval: int = 1,
        log_output: bool = False,
        use_distributed: bool = False,
        verbose: bool = False,
        incremental_grad: bool = False,
        incremental_loss_gap: bool = False,
        incremental_grad_eps: float = 0.001,
        incremental_buffer: int = 5,
        incremental_max_iter: int = 1,
        incremental_grad_max_iter: int = 10,
        incremental_loss_eps: float = 0.001,
    ):
        assert isinstance(model, FNO) or isinstance(
            model, TFNO
        ), f"Error: \
            IncrementalFNOTrainer is designed to work with FNO or TFNO, instead got\
            a model of type {model.__class__.__name__}"

        super().__init__(
            model=model,
            n_epochs=n_epochs,
            wandb_log=wandb_log,
            device=device,
            mixed_precision=mixed_precision,
            data_processor=data_processor,
            eval_interval=eval_interval,
            log_output=log_output,
            use_distributed=use_distributed,
            verbose=verbose,
        )

        self.incremental_loss_gap = incremental_loss_gap
        self.incremental_grad = incremental_grad
        self.incremental = self.incremental_loss_gap or self.incremental_grad
        assert (
            self.incremental
        ), "Error: IncrementalTrainer expects at least one incremental algorithm to be True."
        assert not (
            self.incremental_loss_gap and self.incremental_grad
        ), "Error: IncrementalTrainer expects only one incremental algorithm to be True."

        self.incremental_grad_eps = incremental_grad_eps
        self.incremental_buffer = incremental_buffer
        self.incremental_max_iter = incremental_max_iter
        self.incremental_grad_max_iter = incremental_grad_max_iter
        self.incremental_loss_eps = incremental_loss_eps
        self.loss_list = []

    def incremental_update(self, loss=None):
        """
        Main incremental update function that determines which algorithm to run.

        This method is called after each training epoch to potentially increase the number
        of Fourier modes in the FNO model based on the selected incremental algorithm.

        Parameters
        ----------
        loss : float or torch.Tensor, optional
            Current training loss value. Required for loss gap algorithm.
            If None and loss gap algorithm is enabled, no update will occur.
        """
        if self.incremental_loss_gap and loss is not None:
            self.loss_gap(loss)
        if self.incremental_grad:
            self.grad_explained()

    def train_one_epoch(self, epoch, train_loader, training_loss):
        """
        Train the model for one epoch with incremental learning.

        Extends base trainer by adding incremental learning updates after each epoch.
        May increase Fourier modes based on training progress.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        train_loader : torch.utils.data.DataLoader
            DataLoader containing training data.
        training_loss : callable
            Loss function to use for training.

        Returns
        -------
        tuple
            (train_err, avg_loss, avg_lasso_loss, epoch_train_time)
        """
        self.training = True
        if self.data_processor:
            self.data_processor.epoch = epoch

        # Run base training epoch
        train_err, avg_loss, avg_lasso_loss, epoch_train_time = super().train_one_epoch(
            epoch, train_loader, training_loss
        )

        # Apply incremental learning updates
        self.incremental_update(avg_loss)

        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

    # Algorithm 1: Incremental
    def loss_gap(self, loss):
        """
        Loss gap algorithm for incremental learning.

        Monitors training loss convergence and increases Fourier modes when loss improvement
        becomes too small. Helps escape local minima by increasing model capacity.

        Algorithm:
        1. Track training losses over epochs
        2. Compute difference between consecutive losses
        3. If difference < threshold, increase modes by 1
        4. Update FNO blocks with new mode count

        Parameters
        ----------
        loss : float or torch.Tensor
            Current epoch's training loss value.

        """
        self.loss_list.append(loss)
        self.ndim = len(self.model.fno_blocks.convs[0].n_modes)

        # method 1: loss_gap
        incremental_modes = self.model.fno_blocks.convs[0].n_modes[0]
        max_modes = self.model.fno_blocks.convs[0].max_n_modes[0]
        if len(self.loss_list) > 1:
            loss_difference = abs(self.loss_list[-1] - self.loss_list[-2])
            if loss_difference <= self.incremental_loss_eps:
                # Increase modes if we haven't reached the maximum
                if incremental_modes < max_modes:
                    incremental_modes += 1

        # Update all FNO blocks with the new mode count
        modes_list = tuple([incremental_modes] * self.ndim)
        self.model.fno_blocks.convs[0].n_modes = modes_list

    def grad_explained(self):
        """
        Gradient-based explained variance algorithm for incremental learning.

        Analyzes gradient patterns of FNO weights to determine when additional
        Fourier modes are needed by computing explained variance of gradient strengths.

        Algorithm:
        1. Accumulate gradients over multiple iterations
        2. Compute Frobenius norm of gradients for each Fourier mode
        3. Compute explained variance of gradient strengths
        4. If explained variance < threshold, increase modes
        5. Reset accumulation and update model
        """

        if not hasattr(self, "accumulated_grad"):
            self.accumulated_grad = torch.zeros_like(
                self.model.fno_blocks.convs[0].weight
            )
        if not hasattr(self, "grad_iter"):
            self.grad_iter = 1

        self.ndim = len(self.model.fno_blocks.convs[0].n_modes)

        # Accumulate gradients over multiple iterations
        if self.grad_iter <= self.incremental_grad_max_iter:
            self.grad_iter += 1
            self.accumulated_grad += self.model.fno_blocks.convs[0].weight
        else:
            incremental_final = []

            for i in range(self.ndim):
                max_modes = self.model.fno_blocks.convs[i].max_n_modes[0]
                incremental_modes = self.model.fno_blocks.convs[0].n_modes[0]
                weight = self.accumulated_grad

                # Compute gradient strength for each Fourier mode
                strength_vector = []
                for mode_index in range(min(weight.shape[1], incremental_modes)):
                    strength = torch.norm(weight[:, mode_index, :], p="fro")
                    strength_vector.append(strength)

                # Compute explained variance of gradient strengths
                explained_ratio = compute_explained_variance(
                    incremental_modes - self.incremental_buffer,
                    torch.Tensor(strength_vector),
                )

                # Increase modes if explained variance is too low
                if explained_ratio < self.incremental_grad_eps:
                    if incremental_modes < max_modes:
                        incremental_modes += 1

                incremental_final.append(incremental_modes)

            # update the modes and frequency dimensions
            self.grad_iter = 1
            self.accumulated_grad = torch.zeros_like(self.model.fno_blocks.convs[0].weight)
            main_modes = incremental_final[0]
            modes_list = tuple([main_modes] * self.ndim)
            self.model.fno_blocks.convs[0].n_modes = tuple(modes_list)
