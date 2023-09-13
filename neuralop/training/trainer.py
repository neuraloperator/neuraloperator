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

        if self.callbacks:
            assert type(callbacks) == list, "Callbacks must be a list of Callback objects"
            self.callbacks = callbacks
        else:
            self.callbacks = []
        
        # unless load to device is overriden, call a basic loading function
        override_load_to_device = False
        for callback in self.callbacks:
            if callback.on_load_to_device is not Callback.on_load_to_device:
                self.override_load_to_device = True

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
        
        
    def train(self, train_loader, test_loaders, output_encoder,
            optimizer, scheduler, regularizer, dataset_preprocess_fn,
              training_loss=None, eval_losses=None):
        
        """Trains the given model on the given datasets.
        params:
        train_loader: torch.utils.data.DataLoader
            training dataloader
        test_loader: torch.utils.data.DataLoader
            testing dataloader
        output_encoder: OutputEncoder or None
            encoding/normalizations to perform on model output.
        dataset_preprocess_fn: object, default is None
            function that converts output from dataloader.__getitem__ into form for 
            computation. 
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

                if dataset_preprocess_fn:
                    sample = dataset_preprocess_fn(sample)
                # Decide what to do about logging later when we decide on batch naming conventions
                '''if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f'Training on raw inputs of size {x.shape=}, {y.shape=}')'''

                y = sample.pop('y')

                if self.use_mg_patching:
                    x, y = self.patcher.patch(sample['x'], y)
                    sample['x'] = x
                
                '''
                if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f'.. patched inputs of size {x.shape=}, {y.shape=}')'''

                '''x = x.to(self.device)
                y = y.to(self.device)'''

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

                if self.amp_autocast:
                    with amp.autocast(enabled=True):
                        loss = training_loss(pred=out.float(), truth=y, **sample)
                else:
                    loss = training_loss(pred=out.float(), truth=y, **sample)

                if regularizer:
                    loss += regularizer.loss

                loss.backward()
                
                optimizer.step()
                train_err += loss.item()
        
                with torch.no_grad():
                    avg_loss += loss.item()
                    if regularizer:
                        avg_lasso_loss += regularizer.loss

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_err)
            else:
                scheduler.step()

            epoch_train_time = default_timer() - t1
            del x, y

            

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

                    errors = self.evaluate(eval_losses, loader, output_encoder, log_prefix=loader_name)

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
                    self.callbacks.on_val_batch_start(idx=idx, sample=sample)
                
                y = sample['y']
                n_samples += y.size(0)
                
                if self.use_mg_patching:
                    x, y = self.patcher.patch(sample['x'],y)
                    sample['x'] = x

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
                    errors[f'{log_prefix}_{loss_name}'] += loss(out, y).item()

        del x, y, out

        for key in errors.keys():
            errors[key] /= n_samples

        return errors
