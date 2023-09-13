import torch
from torch.cuda import amp
from timeit import default_timer
import sys 
import wandb

import neuralop.mpu.comm as comm

from .patching import MultigridPatching2D
from .losses import LpLoss
from .callbacks import Callback


class Trainer:
    def __init__(self, *, 
                 model, 
                 n_epochs, 
                 output_field_indices=None, 
                 wandb_log=True, 
                 device=None, 
                 amp_autocast=False, 
                 sample_max = None,
                 callbacks = None,
                 log_test_interval=1, 
                 log_output=False, 
                 use_distributed=False, 
                 verbose=True):
        """
        A general Trainer class to train neural-operators on given datasets

        Parameters
        ----------
        model : nn.Module
        n_epochs : int
        output_field_indices : dict | None
            if a model has multiple output fields, this maps to
            the indices of a model's output associated with each field. 
        wandb_log : bool, default is True
        device : torch.device
        amp_autocast : bool, default is False
        log_test_interval : int, default is 1
            how frequently to print updates
        log_output : bool, default is False
            if True, and if wandb_log is also True, log output images to wandb
        use_distributed : bool, default is False
            whether to use DDP
        verbose : bool, default is True
        """

        if callbacks:
            assert type(callbacks) == list, "Callbacks must be a list of Callback objects"
            self.callbacks = callbacks
        else:
            self.callbacks = []
        
        # unless load to device is overriden, call a basic loading function
        # More than one callback cannot separately overload the device loading
        overrides_device_load = [getattr(c, "on_load_to_device") == getattr(Callback, "on_load_to_device")\
                                  for c in callbacks]
        assert sum(overrides_device_load) < 2, "More than one callback cannot override device loading"
        if sum(overrides_device_load) == 1:
            self.override_load_to_device = True
            print("using custom callback to load data to device.")
        else:
            self.override_load_to_device = False
            print("using standard method to load data to device.")

        # unless loss computation is overriden, call a basic loss function calculation
        overrides_loss = [getattr(c, "compute_training_loss") == getattr(Callback, "compute_training_loss")\
                                  for c in callbacks]
        
        if sum(overrides_loss) == 1:
            self.overrides_loss = True
            print("using custom callback to compute loss.")
        else:
            self.overrides_loss = False
            print("using standard method to compute loss.")

        for callback in self.callbacks:
            callback.on_init_start(model=model, 
                 n_epochs=n_epochs, 
                 wandb_log=wandb_log, 
                 device=device, 
                 amp_autocast=amp_autocast, 
                 log_test_interval=log_test_interval, 
                 log_output=log_output, 
                 use_distributed=use_distributed, 
                 verbose=verbose)

        self.model = model
        self.n_epochs = n_epochs

        if not output_field_indices:
            self.output_field_indices = {'':None}
        else:
            self.output_field_indices = output_field_indices
        self.output_fields = list(self.output_field_indices.keys())

        self.wandb_log = wandb_log
        self.log_test_interval = log_test_interval
        self.log_output = log_output
        self.verbose = verbose
        self.sample_max = sample_max
        self.use_distributed = use_distributed
        self.device = device
        self.amp_autocast = amp_autocast

        for callback in self.callbacks:
            callback.on_init_end(model=model, 
                 n_epochs=n_epochs, 
                 wandb_log=wandb_log, 
                 device=device, 
                 amp_autocast=amp_autocast, 
                 log_test_interval=log_test_interval, 
                 log_output=log_output, 
                 use_distributed=use_distributed, 
                 verbose=verbose)
        
        
    def train(self, train_loader, test_loaders,
            optimizer, scheduler, regularizer,
              training_loss=None, eval_losses=None):
        
        """Trains the given model on the given datasets.
        params:
        train_loader: torch.utils.data.DataLoader
            training dataloader
        test_loader: torch.utils.data.DataLoader
            testing dataloader
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        optimizer: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: function to use 
        """

        for callback in self.callbacks:
            callback.on_train_start(train_loader=train_loader, test_loaders=test_loaders,
                                    optimizer=optimizer, scheduler=scheduler, 
                                    regularizer=regularizer, training_loss=training_loss, 
                                    eval_losses=eval_losses)

        if training_loss is None:
            training_loss = LpLoss(d=2)

        if eval_losses is None: # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)
        
        if self.use_distributed:
            is_logger = (comm.get_world_rank() == 0)
        else:
            is_logger = True 
        
        for epoch in range(self.n_epochs):

            for callback in self.callbacks:
                callback.on_epoch_start(epoch=epoch)

            avg_loss = 0
            avg_lasso_loss = 0
            self.model.train()
            t1 = default_timer()
            train_err = 0.0

            for idx, sample in enumerate(train_loader):

                for callback in self.callbacks:
                    callback.on_batch_start(idx=idx, sample=sample)

                # Decide what to do about logging later when we decide on batch naming conventions
                '''if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f'Training on raw inputs of size {x.shape=}, {y.shape=}')'''

                y = sample['y']

                # load everything from the batch onto self.device if 
                # no callback overrides default load to device
                
                if self.override_load_to_device:
                    for callback in self.callbacks:
                        callback.on_load_to_device(sample=sample)
                else:
                    for k,v in sample.items():
                        if hasattr(v, 'to'):
                            sample[k] = v.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                if regularizer:
                    regularizer.reset()
                
                if self.amp_autocast:
                    with amp.autocast(enabled=True):
                        out = self.model(**sample)
                else:
                    out = self.model(**sample)

                for callback in self.callbacks:
                    callback.on_before_loss(out=out)

                loss = 0.

                if self.overrides_loss:
                    for callback in self.callbacks:
                        if isinstance(out, torch.Tensor):
                            loss += callback.compute_training_loss(out.float(), **sample, amp_autocast=self.amp_autocast)
                        elif isinstance(out, dict):
                            loss += callback.compute_training_loss(**out, **sample, amp_autocast=self.amp_autocast)
                else:
                    if self.amp_autocast:
                        with amp.autocast(enabled=True):
                            if isinstance(out, torch.Tensor):
                                loss = training_loss(out.float(), **sample)
                            elif isinstance(out, dict):
                                loss += training_loss(**out, **sample)
                    else:
                        if isinstance(out, torch.Tensor):
                            loss = training_loss(out.float(), **sample)
                        elif isinstance(out, dict):
                            loss += training_loss(**out, **sample)
                
                del out

                if regularizer:
                    loss += regularizer.loss
                
                loss.backward()
                
                optimizer.step()
                train_err += loss.item()
        
                with torch.no_grad():
                    avg_loss += loss.item()
                    if regularizer:
                        avg_lasso_loss += regularizer.loss

                for callback in self.callbacks:
                    callback.on_batch_end()

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_err)
            else:
                scheduler.step()

            epoch_train_time = default_timer() - t1            

            train_err /= len(train_loader)
            avg_loss  /= self.n_epochs
            
            if epoch % self.log_test_interval == 0: 
                
                msg = f'[{epoch}] time={epoch_train_time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}'

                values_to_log = dict(train_err=train_err, time=epoch_train_time, avg_loss=avg_loss)

                for loader_name, loader in test_loaders.items():
                    if epoch == self.n_epochs - 1 and self.log_output:
                        to_log_output = True
                    else:
                        to_log_output = False

                    errors = self.evaluate(eval_losses, loader, log_prefix=loader_name)

                    for loss_name, loss_value in errors.items():
                        msg += f', {loss_name}={loss_value:.4f}'
                        values_to_log[loss_name] = loss_value

                if regularizer:
                    avg_lasso_loss /= self.n_epochs
                    msg += f', avg_lasso={avg_lasso_loss:.5f}'

                if self.verbose and is_logger:
                    print(msg)
                    sys.stdout.flush()

                # Wandb loging
                if self.wandb_log and is_logger:
                    for pg in optimizer.param_groups:
                        lr = pg['lr']
                        values_to_log['lr'] = lr
                    wandb.log(values_to_log, step=epoch, commit=True)
            
            for callback in self.callbacks:
                callback.on_epoch_end(epoch_train_time=epoch_train_time,
                                      train_err=train_err,
                                      avg_loss=avg_loss,
                                      avg_lasso_loss=avg_lasso_loss)

    def evaluate(self, loss_dict, data_loader,
                 log_prefix=''):
        """Evaluates the model on a dictionary of losses
        
        Parameters
        ----------
        loss_dict : dict of functions 
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary

        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """

        for callback in self.callbacks:
            callback.on_before_val(loss_dict=loss_dict,
                                   data_loader=data_loader)

        self.model.eval()

        if self.use_distributed:
            is_logger = (comm.get_world_rank() == 0)
        else:
            is_logger = True 

        errors = {f'{log_prefix}_{loss_name}':0 for loss_name in loss_dict.keys()}

        n_samples = 0
        with torch.no_grad():
            for idx, sample in enumerate(data_loader):
                
                for callback in self.callbacks:
                    callback.on_val_batch_start(idx=idx, sample=sample)
                
                y = sample['y']
                n_samples += y.size(0)

                # load everything from the batch onto self.device if 
                # no callback overrides default load to device
                
                if self.override_load_to_device:
                    for callback in self.callbacks:
                        callback.on_load_to_device(sample=sample)
                else:
                    for k,v in sample.items():
                        if hasattr(v, 'to'):
                            sample[k] = v.to(self.device)
                
                out = self.model(**sample)

                for callback in self.callbacks:
                    callback.on_before_val_loss(out=out)
                
                for loss_name, loss in loss_dict.items():
                    errors[f'{log_prefix}_{loss_name}'] += loss(out, **sample).item()

        del y, out

        for key in errors.keys():
            errors[key] /= n_samples

        return errors
