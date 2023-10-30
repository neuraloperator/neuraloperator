"""
Callbacks store all non-essential logic
required to run specific training scripts. 

The callbacks in this module follow the form and 
logic of callbacks in Pytorch-Lightning (https://lightning.ai/docs/pytorch/stable)
"""

import os
from pathlib import Path
import sys
from typing import List, Tuple, Union

import torch
import wandb
import numpy as np

from neuralop.training.patching import MultigridPatching2D

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

        overrides_device_load = ["on_load_to_device" in c.__class__.__dict__.keys() for c in callbacks]
       
        assert sum(overrides_device_load) < 2, "More than one callback cannot override device loading"
        if sum(overrides_device_load) == 1:
            self.device_load_callback_idx = overrides_device_load.index(True)
            print("using custom callback to load data to device.")
        else:
            self.device_load_callback_idx = None
            print("using standard method to load data to device.")

        # unless loss computation is overriden, call a basic loss function calculation
        overrides_loss = ["compute_training_loss" in c.__class__.__dict__.keys() for c in callbacks]

        if sum(overrides_loss) >= 1:
            self.overrides_loss = True
            print("using custom callback to compute loss.")
        else:
            self.overrides_loss = False
            print("using standard method to compute loss.")
        
        self.overrides_loss_idx = overrides_loss

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
            self.callbacks[self.device_load_callback_idx].on_load_to_device(*args, *kwargs)
    
    def on_before_forward(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_before_forward(*args, **kwargs)

    def on_before_loss(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_before_loss(*args, **kwargs)
    
    def compute_training_loss(self, *args, **kwargs):
        loss = 0.
        if self.overrides_loss:
            for i,c in enumerate(self.callbacks):
                if self.overrides_loss_idx[i]:
                    loss += c.compute_training_loss(*args, **kwargs)
            return loss
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
    
    def compute_val_loss(self, *args, **kwargs):
        loss = 0.
        if self.overrides_loss:
            for i,c in enumerate(self.callbacks):
                if self.overrides_loss_idx[i]:
                    c.compute_val_loss(*args, **kwargs)
            return loss
        else:
            pass
    
    def on_val_batch_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_val_batch_end(*args, **kwargs)

    def on_val_epoch_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_val_epoch_end(*args, **kwargs)
    
    def on_val_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_val_end(*args, **kwargs)

class SimpleWandBLoggerCallback(Callback):
    """
    Callback that implements simple logging functionality 
    expected when passing verbose to a Trainer
    """

    def __init__(self, **kwargs):
        super().__init__()
        if kwargs:
            wandb.init(**kwargs)
    
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
    
    def on_batch_start(self, idx, **kwargs):
        self._update_state_dict(idx=idx)

    def on_before_loss(self, out, **kwargs):
        if self.state_dict['epoch'] == 0 and self.state_dict['idx'] == 0 \
            and self.state_dict['verbose']:
            print(f'Raw outputs of size {out.shape=}')
    
    def on_before_val(self, epoch, train_err, time, avg_loss, avg_lasso_loss, **kwargs):
        # track training err and val losses to print at interval epochs
        msg = f'[{epoch}] time={time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}'
        values_to_log = dict(train_err=train_err / self.state_dict['n_train'], time=time, avg_loss=avg_loss)

        self._update_state_dict(msg=msg, values_to_log=values_to_log)
        self._update_state_dict(avg_lasso_loss=avg_lasso_loss)
        
    def on_val_epoch_end(self, errors, **kwargs):
        for loss_name, loss_value in errors.items():
            self.state_dict['msg'] += f', {loss_name}={loss_value:.4f}'
            self.state_dict['values_to_log'][loss_name] = loss_value
    
    def on_val_end(self, *args, **kwargs):
        if self.state_dict.get('regularizer', False):
            avg_lasso = self.state_dict.get('avg_lasso_loss', 0.)
            avg_lasso /= self.state_dict.get('n_epochs')
            self.state_dict['msg'] += f', avg_lasso={avg_lasso:.5f}'
        
        print(self.state_dict['msg'])
        sys.stdout.flush()

        wandb_log = self.state_dict.get('wandb_log', False)
        if self.state_dict.get('wandb_log', False):
            for pg in self.state_dict['optimizer'].param_groups:
                lr = pg['lr']
                self.state_dict['values_to_log']['lr'] = lr
            wandb.log(self.state_dict['values_to_log'], step=self.state_dict['epoch'], commit=True)

class MGPatchingCallback(Callback):
    def __init__(self, levels: int, padding_fraction: float, stitching: float, encoder=None):
        """MGPatchingCallback implements multigrid patching functionality
        for datasets that require domain patching, stitching and/or padding.

        Parameters
        ----------
        levels : int
            mg_patching level parameter for MultigridPatching2D
        padding_fraction : float
            mg_padding_fraction parameter for MultigridPatching2D
        stitching : _type_
            mg_patching_stitching parameter for MultigridPatching2D
        encoder : neuralop.datasets.output_encoder.OutputEncoder, optional
            OutputEncoder to decode model outputs, by default None
        """
        super().__init__()
        self.levels = levels
        self.padding_fraction = padding_fraction
        self.stitching = stitching
        self.encoder = encoder
        
    def on_init_end(self, **kwargs):
        self._update_state_dict(**kwargs)
        self.patcher = MultigridPatching2D(model=self.state_dict['model'], levels=self.levels, 
                                      padding_fraction=self.padding_fraction,
                                      stitching=self.stitching)
    
    def on_batch_start(self, **kwargs):
        self._update_state_dict(**kwargs)
        self.state_dict['sample']['x'],self.state_dict['sample']['y'] =\
              self.patcher.patch(self.state_dict['sample']['x'],
                                 self.state_dict['sample']['y'],)
    
    def on_val_batch_start(self, *args, **kwargs):
        return self.on_batch_start(*args, **kwargs)
        
    def on_before_loss(self, out, **kwargs):
        
        evaluation = kwargs.get('eval', False)
        self._update_state_dict(out=out)
        self.state_dict['out'], self.state_dict['sample']['y'] = \
            self.patcher.unpatch(self.state_dict['out'],
                                 self.state_dict['sample']['y'],
                                 evaluation=evaluation)

        if self.encoder:
            self.state_dict['out'] = self.encoder.decode(self.state_dict['out'])
            self.state_dict['sample']['y'] = self.encoder.decode(self.state_dict['sample']['y'])
        
    
    def on_before_val_loss(self, **kwargs):
        return self.on_before_loss(**kwargs, evaluation=True)


class OutputEncoderCallback(Callback):
    
    def __init__(self, encoder):
        """
        Callback class for a training loop that involves
        an output normalizer but no MG patching.

        Parameters
        -----------
        encoder : neuralop.datasets.output_encoder.OutputEncoder
            module to normalize model inputs/outputs
        """
        super().__init__()
        self.encoder = encoder
    
    def on_batch_start(self, *args, **kwargs):
        self._update_state_dict(**kwargs)
    
    def on_before_loss(self, out):
        self.state_dict['out'] = self.encoder.decode(out)
        self.state_dict['sample']['y'] = self.encoder.decode(self.state_dict['sample']['y'])
    
    def on_before_val_loss(self, **kwargs):
        return self.on_before_loss(**kwargs)

class TransformCallback(Callback):
    
    def __init__(self, transform):
        """
        Callback class for a training loop that involves
        an output normalizer but no MG patching.

        Parameters
        -----------
        encoder : neuralop.datasets.output_encoder.OutputEncoder
            module to normalize model inputs/outputs
        """
        super().__init__()
        self.transform = transform
    
    def on_batch_start(self, *args, **kwargs):
        self._update_state_dict(**kwargs)
    
    def on_before_loss(self, out):
        self.state_dict['out'] = self.transform.inverse_transform(out)
        self.state_dict['sample']['y'] = self.transform.inverse_transform(self.state_dict['sample']['y'])
    
    def on_before_val_loss(self, **kwargs):
        return self.on_before_loss(**kwargs)
        
class CheckpointCallback(Callback):
    
    def __init__(self, 
                 save_dir: Union[Path, str], 
                 save_best : str = None,
                 save_interval : int = 1,
                 save_optimizer : bool = False,
                 save_scheduler : bool = False,
                 save_regularizer : bool = False,
                 resume_from_dir : Union[Path, str] = None
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
        

    def on_init_end(self, *args, **kwargs):
        self._update_state_dict(**kwargs)

    
    def on_train_start(self, *args, **kwargs):
        self._update_state_dict(**kwargs)

        verbose = self.state_dict.get('verbose', False)
        if self.save_best:
            assert self.state_dict['eval_losses'], "Error: cannot monitor a metric if no validation metrics exist."
            assert self.save_best in self.state_dict['eval_losses'].keys(), "Error: cannot monitor a metric outside of eval_losses."
            self.best_metric_value = float('inf')
        else:
            self.best_metric_value = None
        
        # load state dict if resume_from_dir is given
        if self.resume_from_dir:
            saved_modules = [x.stem for x in self.resume_from_dir.glob('*.pt')]

            assert 'best_model' in saved_modules or 'model' in saved_modules,\
                  "Error: CheckpointCallback expects a model state dict named model.pt or best_model.pt."
            
            # no need to handle exceptions if assertion that either model file exists passes
            if 'best_model' in saved_modules:
                self.state_dict['model'].load_state_dict(torch.load(self.resume_from_dir / 'best_model.pt'))
                if verbose:
                    print(f"Loading model state from best_model.pt")
            else:
                self.state_dict['model'].load_state_dict(torch.load(self.resume_from_dir / 'model.pt'))
                if verbose:
                    print(f"Loading model state from model.pt")
            
            # load all of optimizer, scheduler, regularizer if they exist
            for module in ['optimizer', 'scheduler', 'regularizer']:
                if module in saved_modules:
                    self.state_dict[module].load_state_dict(torch.load(self.resume_from_dir / f"{module}.pt"))

    def on_epoch_start(self, *args, **kwargs):
        self._update_state_dict(**kwargs)
    
    def on_val_epoch_start(self, *args, **kwargs):
        self._update_state_dict(**kwargs)

    def on_val_epoch_end(self, *args, **kwargs):
        """
        Update state dict with errors
        """
        self._update_state_dict(**kwargs)

    def on_epoch_end(self, *args, **kwargs):
        """
        Save state to dir if all conditions are met
        """
        if self.save_best:
            log_prefix = self.state_dict['log_prefix']
            if self.state_dict['errors'][f"{log_prefix}_{self.save_best}"] < self.best_metric_value:
                metric_cond = True
            else:
                metric_cond = False
        else:
            metric_cond=True

        # Save states to save_dir 
        if self.state_dict['epoch'] % self.save_interval == 0 and metric_cond:
            # save model or best_model.pt no matter what
            if self.save_best:
                model_name = 'best_model'
            else:
                model_name = 'model'
            save_path = self.save_dir / f"{model_name}.pt"
            torch.save(self.state_dict['model'].state_dict(), save_path)

            # save optimizer, scheduler, regularizer according to flags
            if self.save_optimizer:
                save_path = self.save_dir / "optimizer.pt"
                torch.save(self.state_dict['optimizer'].state_dict(), save_path)
            if self.save_scheduler:
                save_path = self.save_dir / "scheduler.pt"
                torch.save(self.state_dict['scheduler'].state_dict(), save_path)
            if self.save_regularizer:
                save_path = self.save_dir / "regularizer.pt"
                torch.save(self.state_dict['regularizer'].state_dict(), save_path)
            
            if self.state_dict['verbose']:
                print(f"Saved training state to {save_path}")

class RegularDatasetToFNOGNOCallback(Callback):
    def __init__(self, out_p_gradient: bool=False, 
                 pad: int=0,
                 include_out_p_endpoint: bool=True,
                 domain_lengths: Union[Union[Tuple[float], List[float]], float] = 1):
        """RegularDatasetToFNOGNOCallback

        Performs simple data preprocessing to convert a data batch
        from a form expected by an FNO or similar model to that
        expected by an FNOGNO.

        Params
        -------
        out_p_gradient: bool, defaults to False
            whether generated output grid requires_grad or not
        pad: int, defaults to 0.
            number of pixels/discretization units to pad on each side
        include_out_p_endpoint: bool, defaults to True
            whether to include the endpoint of out_p
        domain_lengths: List or Tuple of float, or float
            size of data domain along each dimension

        TODO @COLIN: should this belong in datasets
        under some "generate_out_p=True" flag?
        """
        super().__init__()
        self.out_p_gradient = out_p_gradient
        self.pad = pad
        self.include_out_p_endpoint = include_out_p_endpoint
        self.domain_lengths=domain_lengths
    
    def create_out_p(self, batch_size=16, train_resolution=(16,16), device=None, sample_random=None, bound_dist=0):
        domain_lengths = self.domain_lengths
        if not isinstance(self.domain_lengths, (tuple, list)):
            domain_lengths = [domain_lengths] * len(train_resolution)

        if sample_random is None:
            include_endpoint = self.include_out_p_endpoint
            if not isinstance(include_endpoint, (tuple, list)):
                include_endpoint = [include_endpoint] * len(train_resolution)

            tx = []
            for resolution, domain_length, endpoint in zip(train_resolution, domain_lengths, include_endpoint):
                if endpoint:
                    tx.append(torch.linspace(0, domain_length, resolution))
                else:
                    tx.append(torch.linspace(0, domain_length, resolution + 1)[:-1])

            out_p = torch.stack(
                torch.meshgrid(*tx, indexing="ij"), axis=-1
            ).astype(torch.float32)

            # duplicate batch_size times
            out_p = torch.tile(out_p, (batch_size, 1, 1, 1))
            # convert to a list of coordinate points
            out_p = out_p.reshape(batch_size, out_p.shape[1] * out_p.shape[2], 2)
            out_p = out_p.to(device)
        
        else:
            # TODO(jberner): make this more general for other boundary conditions
            x = np.linspace(0, domain_lengths[1], train_resolution[1] + 1)[:-1]
            x = torch.from_numpy(x).float().to(device)
            boundary = torch.stack([torch.zeros_like(x), x], dim=1).tile([batch_size, 1, 1])

            domain_lengths = torch.tensor(domain_lengths, device=device)
            out_p = bound_dist + torch.rand(batch_size, sample_random, len(train_resolution), device=device) * (domain_lengths - 2 * bound_dist)
            out_p = torch.cat([out_p, boundary], dim=1)
        return out_p

    
    def on_batch_start(self, *args, **kwargs):
        self._update_state_dict(**kwargs)
        x = self.state_dict['sample']['x']
        train_resolution = [s - 2 * self.pad for s in x.shape[-2:]]
        out_p = self.create_out_p(batch_size=x.shape[0], train_resolution=train_resolution, device=self.device, sample_random=self.sample_random)
        out_p.requires_grad_(self.out_p_gradient)

