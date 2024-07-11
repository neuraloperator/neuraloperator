"""
Callbacks store all non-essential logic
required to run specific training scripts. 

The callbacks in this module follow the form and 
logic of callbacks in Pytorch-Lightning (https://lightning.ai/docs/pytorch/stable)
"""

import os
from pathlib import Path
import sys
from typing import List, Union, Literal

import torch
import wandb

from .training_state import save_training_state, load_training_state
from neuralop.utils import compute_rank, compute_stable_rank, compute_explained_variance


class Callback(object):
    """
    Base callback class. Each abstract method is called in the trainer's
    training loop at the appropriate time.

    Callbacks are stateful, meaning they keep track of a state and
        update it throughout the lifetime of a Trainer class.
        Storing the state as a dict enables the Callback to keep track of
        references to underlying parts of the Trainer's process, such as
        models, cost functions and output encoders
    """

    def __init__(self):
        self.state_dict = {}

    def _update_state_dict(self, **kwargs):
        self.state_dict.update(kwargs)

    def on_init_start(self, **kwargs):
        pass

    def on_init_end(self, *args, **kwargs):
        pass

    def on_before_train(self, *args, **kwargs):
        pass

    def on_train_start(self, *args, **kwargs):
        pass

    def on_epoch_start(self, *args, **kwargs):
        pass

    def on_batch_start(self, *args, **kwargs):
        pass

    def on_load_to_device(self, *args, **kwargs):
        pass

    def on_before_forward(self, *args, **kwargs):
        pass

    def on_before_loss(self, *args, **kwargs):
        pass

    def compute_training_loss(self, *args, **kwargs):
        raise NotImplementedError

    def on_batch_end(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        pass

    def on_train_end(self, *args, **kwargs):
        pass

    def on_before_val(self, *args, **kwargs):
        pass

    def on_val_epoch_start(self, *args, **kwargs):
        pass

    def on_val_batch_start(self, *args, **kwargs):
        pass

    def on_before_val_loss(self, **kwargs):
        pass

    def compute_val_loss(self, *args, **kwargs):
        pass

    def on_val_batch_end(self, *args, **kwargs):
        pass

    def on_val_epoch_end(self, *args, **kwargs):
        pass

    def on_val_end(self, *args, **kwargs):
        pass


class IncrementalCallback(Callback):
    """
    Callback that implements the Incremental Algorithm - Both the Gradient explained and Loss Gap versions
    
    incremental : bool, default is False
        if True, use the base incremental algorithm which is based on gradient variance
        uses the incremental_grad_eps parameter - set the threshold for gradient variance
        uses the incremental_buffer paramater - sets the number of buffer modes to calculate the gradient variance
        uses the incremental_max_iter parameter - sets the initial number of iterations
        uses the incremental_grad_max_iter parameter - sets the maximum number of iterations to accumulate the gradients
    incremental_loss_gap : bool, default is False
        if True, use the incremental algorithm based on loss gap
        uses the incremental_loss_eps parameter
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
        assert self.incremental, "Error: IncrementalCallback expects at least one incremental algorithm to be True."
        assert not (self.incremental_loss_gap and self.incremental_grad), "Error: IncrementalCallback expects only one incremental algorithm to be True."
        
        self.incremental_grad_eps = incremental_grad_eps
        self.incremental_buffer = incremental_buffer
        self.incremental_max_iter = incremental_max_iter
        self.incremental_grad_max_iter = incremental_grad_max_iter
        self.incremental_loss_eps = incremental_loss_eps
        self.loss_list = []
        self.mode = "Train"
    
    def on_init_end(self, *args, **kwargs):
        self._update_state_dict(**kwargs)
    
    def on_train_start(self, **kwargs):
        self._update_state_dict(**kwargs)

        train_loader = self.state_dict['train_loader']
        test_loaders = self.state_dict['test_loaders']
        verbose = self.state_dict['verbose']

        n_train = len(train_loader.dataset)
        self._update_state_dict(n_train=n_train)

        if not isinstance(test_loaders, dict):
            test_loaders = dict(test=test_loaders)

        if verbose:
            print(f'Training on {n_train} samples')
            print(f'Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples'
                  f'         on resolutions {[name for name in test_loaders]}.')
            sys.stdout.flush()
        
    def on_epoch_start(self, epoch):
        self._update_state_dict(epoch=epoch)
        
    def on_epoch_end(self, epoch, **kwargs):
        self._update_state_dict(epoch=epoch)
        print(f'Currently the model is using incremental_n_modes = {self.state_dict["model"].fno_blocks.convs.n_modes}')
    
    def on_batch_start(self, idx, **kwargs):
        self._update_state_dict(idx=idx)
        self.mode = "Train"
        self.data = self.state_dict['data_processor']
        if self.data is not None:
            self.data.epoch = self.state_dict['epoch']
        
    def on_before_loss(self, out, **kwargs):
        if self.state_dict['epoch'] == 0 and self.state_dict['idx'] == 0 \
            and self.state_dict['verbose']:
            print(f'Raw outputs of size {out.shape=}')
    
    def on_before_val(self, epoch, train_err, time, avg_loss, avg_lasso_loss, **kwargs):
        # track training err and val losses to print at interval epochs
        msg = f'[{epoch}] time={time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}'

        self.step(avg_loss)
        
        self._update_state_dict(msg=msg)
        self._update_state_dict(avg_lasso_loss=avg_lasso_loss)
        
    def on_val_epoch_end(self, errors, **kwargs):
        for loss_name, loss_value in errors.items():
            if isinstance(loss_value, float):
                self.state_dict['msg'] += f', {loss_name}={loss_value:.4f}'
            else:
                loss_value = {i:e.item() for (i, e) in enumerate(loss_value)}
                self.state_dict['msg'] += f', {loss_name}={loss_value}'
    
    def on_val_batch_start(self, *args, **kwargs):
        self.mode = "Validation"
        if self.data is not None:
            self.data.epoch = self.state_dict['epoch']

    def on_val_end(self, *args, **kwargs):
        if self.state_dict.get('regularizer', False):
            avg_lasso = self.state_dict.get('avg_lasso_loss', 0.)
            avg_lasso /= self.state_dict.get('n_epochs')
            self.state_dict['msg'] += f', avg_lasso={avg_lasso:.5f}'
        
        print(self.state_dict['msg'])
        sys.stdout.flush()
    
    # Main step function: which algorithm to run
    def step(self, loss=None):
        if self.incremental_loss_gap and loss is not None:
            self.loss_gap(loss)
        if self.incremental_grad:
            self.grad_explained()
    
    # Algorithm 1: Incremental
    def loss_gap(self, loss):
        self.loss_list.append(loss)
        self.ndim = len(self.state_dict['model'].fno_blocks.convs.n_modes)
        # method 1: loss_gap
        incremental_modes = self.state_dict['model'].fno_blocks.convs.n_modes[0]
        max_modes = self.state_dict['model'].fno_blocks.convs.max_n_modes[0]
        if len(self.loss_list) > 1:
            if abs(self.loss_list[-1] - self.loss_list[-2]) <= self.incremental_loss_eps:
                if incremental_modes < max_modes:
                    incremental_modes += 1
        modes_list = tuple([incremental_modes] * self.ndim)
        self.state_dict['model'].fno_blocks.convs.n_modes = modes_list

    # Algorithm 2: Gradient based explained ratio
    def grad_explained(self):
        # for mode 1
        if not hasattr(self, 'accumulated_grad'):
            self.accumulated_grad = torch.zeros_like(
                self.state_dict['model'].fno_blocks.convs.weight[0])
        if not hasattr(self, 'grad_iter'):
            self.grad_iter = 1
            
        self.ndim = len(self.state_dict['model'].fno_blocks.convs.n_modes)
        if self.grad_iter <= self.incremental_grad_max_iter:
            self.grad_iter += 1
            self.accumulated_grad += self.state_dict['model'].fno_blocks.convs.weight[0]
        else:
            incremental_final = []
            for i in range(self.ndim):
                max_modes = self.state_dict['model'].fno_blocks.convs.max_n_modes[i]
                incremental_modes = self.state_dict['model'].fno_blocks.convs.n_modes[i]
                weight = self.accumulated_grad
                strength_vector = []
                for mode_index in range(
                        min(weight.shape[1], incremental_modes)):
                    strength = torch.norm(
                        weight[:, mode_index, :], p='fro')
                    strength_vector.append(strength)
                expained_ratio = compute_explained_variance(
                    incremental_modes - self.incremental_buffer, torch.Tensor(strength_vector))
                if expained_ratio < self.incremental_grad_eps:
                    if incremental_modes < max_modes:
                        incremental_modes += 1
                incremental_final.append(incremental_modes)

            # update the modes and frequency dimensions
            self.grad_iter = 1
            self.accumulated_grad = torch.zeros_like(
                self.state_dict['model'].fno_blocks.convs.weight[0])
            main_modes = incremental_final[0]
            modes_list = tuple([main_modes] * self.ndim)
            self.state_dict['model'].fno_blocks.convs.n_modes = tuple(modes_list)
