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

from neuralop.losses.meta_losses import WeightedSumLoss


class Callback(object):
    """
    Base callback class. Each abstract method is called in the trainer's
    training loop at the appropriate time.
    """

    def __init__(self):
        pass

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


class PipelineCallback(Callback):
    def __init__(self, callbacks: List[Callback]):
        """
        PipelineCallback handles logic for the case in which
        a user passes more than one Callback to a trainer.

        Parameters
        ----------
        callbacks : List[Callback]
            list of Callbacks to use in Trainer
        """
        self.callbacks = callbacks

        overrides_device_load = [
            "on_load_to_device" in c.__class__.__dict__.keys() for c in callbacks
        ]

        assert (
            sum(overrides_device_load) < 2
        ), "More than one callback cannot override device loading"
        if sum(overrides_device_load) == 1:
            self.device_load_callback_idx = overrides_device_load.index(True)
            print("using custom callback to load data to device.")
        else:
            self.device_load_callback_idx = None
            print("using standard method to load data to device.")

        # unless loss computation is overriden, call a basic loss function calculation
        overrides_train_loss = [
            "compute_training_loss" in c.__class__.__dict__.keys() for c in callbacks
        ]

        assert (
            sum(overrides_train_loss) < 2
        ), "More than one callback cannot override train loss computation"
        if sum(overrides_train_loss) == 1:
            self.loss_callback_idx = overrides_train_loss.index(True)
            print("using custom callback to compute train loss.")
        else:
            self.loss_callback_idx = None
            print("using standard method to compute train loss.")
        self.overrides_train_loss = overrides_train_loss

    def _update_state_dict(self, **kwargs):
        for c in self.callbacks:
            c._update_state_dict(kwargs)

    def on_init_start(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_init_start(*args, **kwargs)

    def on_init_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_init_end(*args, **kwargs)

    def on_before_train(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_before_train(*args, **kwargs)

    def on_train_start(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_train_start(*args, **kwargs)

    def on_epoch_start(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_epoch_start(*args, **kwargs)

    def on_batch_start(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_batch_start(*args, **kwargs)

    def on_load_to_device(self, *args, **kwargs):
        if self.device_load_callback_idx:
            self.callbacks[self.device_load_callback_idx].on_load_to_device(
                *args, *kwargs
            )

    def on_before_forward(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_before_forward(*args, **kwargs)

    def on_before_loss(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_before_loss(*args, **kwargs)

    def compute_training_loss(self, *args, **kwargs):
        if self.overrides_train_loss:
            return self.callbacks[self.loss_callback_idx].compute_training_loss(*args, **kwargs)
        else:
            pass

    def on_batch_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_batch_end(*args, **kwargs)

    def on_epoch_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_epoch_end(*args, **kwargs)

    def on_train_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_train_end(*args, **kwargs)

    def on_before_val(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_before_val(*args, **kwargs)

    def on_val_epoch_start(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_val_epoch_start(*args, **kwargs)

    def on_val_batch_start(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_val_batch_start(*args, **kwargs)

    def on_before_val_loss(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_before_val_loss(*args, **kwargs)

    '''def compute_val_loss(self, *args, **kwargs):
        if self.overrides_loss:
            for c in self.callbacks:
                c.compute_val_loss(*args, **kwargs)
        else:
            pass'''

    def on_val_batch_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_val_batch_end(*args, **kwargs)

    def on_val_epoch_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_val_epoch_end(*args, **kwargs)

    def on_val_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_val_end(*args, **kwargs)


class CheckpointCallback(Callback):
    def __init__(
        self,
        save_dir: Union[Path, str],
        save_best: str = None,
        save_interval: int = 1,
        save_optimizer: bool = False,
        save_scheduler: bool = False,
        save_regularizer: bool = False,
        resume_from_dir: Union[Path, str] = None,
    ):
        """CheckpointCallback handles saving and resuming
        training state from checkpoint .pt save files.

        Parameters
        ----------
        save_dir : Union[Path, str], optional
            folder in which to save checkpoints, by default './checkpoints'
        save_best : str, optional
            metric to monitor for best value in order to save state
        save_interval : int, optional
            interval on which to save/check metric, by default 1
        save_optimizer : bool, optional
            whether to save optimizer state, by default False
        save_scheduler : bool, optional
            whether to save scheduler state, by default False
        save_regularizer : bool, optional
            whether to save regularizer state, by default False
        resume_from_dir : Union[Path, str], optional
            folder from which to resume training state.
            Expects saved states in the form: (all but model optional)
               (best_model.pt or model.pt), optimizer.pt, scheduler.pt, regularizer.pt
            All state files present will be loaded.
            if some metric was monitored during checkpointing,
            the file name will be best_model.pt.
        """

        super().__init__()
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        self.save_dir = save_dir

        self.save_interval = save_interval
        self.save_best = save_best
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.save_regularizer = save_regularizer

        if resume_from_dir:
            if isinstance(resume_from_dir, str):
                resume_from_dir = Path(resume_from_dir)
            assert resume_from_dir.exists()

        self.resume_from_dir = resume_from_dir


    def on_train_start(self, model,
                        train_loader,
                        test_loaders,
                        optimizer,
                        scheduler,
                        regularizer,
                        training_loss,
                        eval_losses,
                        data_processor):

        verbose = self.state_dict.get("verbose", False)
        if self.save_best:
            assert self.state_dict[
                "eval_losses"
            ], "Error: cannot monitor a metric if no validation metrics exist."
            assert (
                self.save_best in self.state_dict["eval_losses"].keys()
            ), "Error: cannot monitor a metric outside of eval_losses."
            self.best_metric_value = float("inf")
        else:
            self.best_metric_value = None

        # load state dict if resume_from_dir is given
        if self.resume_from_dir:
            # check for save model exists
            if (self.resume_from_dir / "best_model_state_dict.pt").exists():
                save_name = "best_model"
            elif (self.resume_from_dir / "model_state_dict.pt").exists():
                save_name = "model"
            else:
                raise FileNotFoundError("Error: CheckpointCallback expects a model\
                                         state dict named model.pt or best_model.pt.")
            # returns key-value pairs "model":model, "optimizer":optimizer...
            training_state = load_training_state(save_dir=self.resume_from_dir, save_name=save_name,
                                                 model=model,
                                                 optimizer=optimizer,
                                                 regularizer=regularizer,
                                                 scheduler=scheduler)
            # reassign references
            model = training_state['model']
            if training_state.get('optimizer', False):
                optimizer = training_state['optimizer']
            if training_state.get('scheduler', False):
                scheduler = training_state['scheduler']
            if training_state.get('regularizer', False):
                regularizer = training_state['regularizer']
            
    

    def on_epoch_end(self, epoch, 
                    train_err, 
                    avg_loss,
                    errors,
                    model):
        """
        Save state to dir if all conditions are met
        """
        if self.save_best:
            log_prefix = self.state_dict["log_prefix"]
            if (
                self.state_dict["errors"][f"{log_prefix}_{self.save_best}"]
                < self.best_metric_value
            ):
                metric_cond = True
            else:
                metric_cond = False
        else:
            metric_cond = True

        # Save states to save_dir
        if self.state_dict["epoch"] % self.save_interval == 0 and metric_cond:
            # save model or best_model.pt no matter what
            if self.save_best:
                model_name = "best_model"
            else:
                model_name = "model"

            save_training_state(
                self.save_dir,
                model_name,
                model=self.state_dict["model"],
                optimizer=self.state_dict.get("optimizer", None),
                regularizer=self.state_dict.get("regularizer", None),
                scheduler=self.state_dict.get("scheduler", None),
            )

            if self.state_dict["verbose"]:
                print(f"Saved training state to {self.save_dir}")

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
        
    def on_epoch_end(self, epoch, **kwargs):
        self._update_state_dict(epoch=epoch)
        print(f'Currently the model is using incremental_n_modes = {self.state_dict["model"].fno_blocks.convs.n_modes}')
    
    def on_batch_start(self, epoch, idx, sample, data_processor):
        self.mode = "Train"
        if data_processor:
            data_processor.epoch = epoch
    
    #TODO: step on loss
    def on_val_batch_start(self, *args, **kwargs):

        self.mode = "Validation"
        if self.data is not None:
            self.data.epoch = self.state_dict['epoch']
    
    # Main step function: which algorithm to run
    def step(self, model, loss=None):
        if self.incremental_loss_gap and loss is not None:
            self.loss_gap(loss, model)
        if self.incremental_grad:
            self.grad_explained(model)
    
    # Algorithm 1: Incremental
    def loss_gap(self, loss, model):
        self.loss_list.append(loss)
        self.ndim = len(model.fno_blocks.convs.n_modes)
        # method 1: loss_gap
        incremental_modes = model.fno_blocks.convs.n_modes[0]
        max_modes = model.fno_blocks.convs.max_n_modes[0]
        if len(self.loss_list) > 1:
            if abs(self.loss_list[-1] - self.loss_list[-2]) <= self.incremental_loss_eps:
                if incremental_modes < max_modes:
                    incremental_modes += 1
        modes_list = tuple([incremental_modes] * self.ndim)
        model.fno_blocks.convs.n_modes = modes_list

    # Algorithm 2: Gradient based explained ratio
    def grad_explained(self, model):
        # for mode 1
        if not hasattr(self, 'accumulated_grad'):
            self.accumulated_grad = torch.zeros_like(
                model.fno_blocks.convs.weight[0])
        if not hasattr(self, 'grad_iter'):
            self.grad_iter = 1
            
        self.ndim = len(model.fno_blocks.convs.n_modes)
        if self.grad_iter <= self.incremental_grad_max_iter:
            self.grad_iter += 1
            self.accumulated_grad += model.fno_blocks.convs.weight[0]
        else:
            incremental_final = []
            for i in range(self.ndim):
                max_modes = model.fno_blocks.convs.max_n_modes[i]
                incremental_modes = model.fno_blocks.convs.n_modes[i]
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
                model.fno_blocks.convs.weight[0])
            main_modes = incremental_final[0]
            modes_list = tuple([main_modes] * self.ndim)
            model.fno_blocks.convs.n_modes = tuple(modes_list)

class LossBalancingCallback(Callback):
    # source: https://arxiv.org/pdf/2308.08468
    # Wang et al., "An Expert's Guide to Training Physics-Informed Neural Networks"
    def __init__(self, alpha: float=0.9):
        super().__init__()
        self.alpha = alpha
    
    def compute_training_loss(self, out, sample, loss, model, **kwargs):
        loss_vals = loss(out, **sample) # sumlossoutput
        grad_norms = {}
        total_grad_norm = 0.
        for name, val in loss_vals.losses.items():
            grads = torch.autograd.grad(outputs=val, inputs=model.parameters(), retain_graph=True)
            full_norm = torch.norm(torch.stack([torch.norm(x, p=2) for x in grads]), p=2)
            grad_norms[name] = full_norm
            total_grad_norm += full_norm
        for name in loss_vals.losses.keys():
            loss_vals.weights[name] = loss_vals.weights[name] * self.alpha +\
            (1-self.alpha) * grad_norms[name] / total_grad_norm
            
        return loss_vals
        