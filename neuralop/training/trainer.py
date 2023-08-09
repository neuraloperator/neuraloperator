import torch
from torch.cuda import amp
from timeit import default_timer
import wandb
import sys 

import neuralop.mpu.comm as comm

from .patching import MultigridPatching2D
from .losses import LpLoss
from .algo import Incremental


class Trainer:
    def __init__(self, model, n_epochs, wandb_log=True, device=None, amp_autocast=False,
                 mg_patching_levels=0, mg_patching_padding=0, mg_patching_stitching=True,
                 log_test_interval=1, log_output=False, use_distributed=False, verbose=True, incremental=False, 
                 incremental_loss_gap=False, incremental_resolution=False, incremental_eps = 0.999, incremental_buffer=5,
                 incremental_max_iter=1, incremental_grad_max_iter=10, incremental_loss_eps=0.01, incremental_res_gap=100, 
                 dataset_resolution=16, dataset_sublist=[1], dataset_indices=[0]):
        """
        A general Trainer class to train neural-operators on given datasets

        Parameters
        ----------
        model : nn.Module
        n_epochs : int
        wandb_log : bool, default is True
        device : torch.device
        amp_autocast : bool, default is False
        mg_patching_levels : int, default is 0
            if 0, no multi-grid domain decomposition is used
            if > 0, indicates the number of levels to use
        mg_patching_padding : float, default is 0
            value between 0 and 1, indicates the fraction of size to use as padding on each side
            e.g. for an image of size 64, padding=0.25 will use 16 pixels of padding on each side
        mg_patching_stitching : bool, default is True
            if False, the patches are not stitched back together and the loss is instead computed per patch
        log_test_interval : int, default is 1
            how frequently to print updates
        log_output : bool, default is False
            if True, and if wandb_log is also True, log output images to wandb
        use_distributed : bool, default is False
            whether to use DDP
        verbose : bool, default is True
        incremental : bool, default is False
            if True, use the base incremental algorithm which is based on gradient variance
            uses the incremental_grad_eps parameter - set the threshold for gradient variance
            uses the incremental_buffer paramater - sets the number of buffer modes to calculate the gradient variance
            uses the incremental_max_iter parameter - sets the initial number of iterations
            uses the incremental_grad_max_iter parameter - sets the maximum number of iterations to accumulate the gradients
        incremental_loss_gap : bool, default is False
            if True, use the incremental algorithm based on loss gap
            uses the incremental_loss_eps parameter
        incremental_resolution : bool, default is False
            if True, increase the resolution of the input incrementally
            uses the incremental_res_gap parameter
            uses the dataset_sublist parameter - a list of resolutions to use
            uses the dataset_indices parameter - a list of indices of the dataset to slice to regularize the input resolution
            uses the dataset_resolution parameter - the resolution of the input
        """
        self.n_epochs = n_epochs
        self.wandb_log = wandb_log
        self.log_test_interval = log_test_interval
        self.log_output = log_output
        self.verbose = verbose
        self.mg_patching_levels = mg_patching_levels
        self.mg_patching_stitching = mg_patching_stitching
        self.use_distributed = use_distributed
        self.device = device
        self.incremental_loss_gap = incremental_loss_gap
        self.incremental_grad = incremental
        self.incremental_resolution = incremental_resolution
        self.incremental = self.incremental_loss_gap or self.incremental_grad
        self.amp_autocast = amp_autocast
        self.grad_eps = incremental_eps
        self.loss_eps = incremental_loss_eps
        self.res_gap = incremental_res_gap
        self.sublist = dataset_sublist
        self.dataset_resolution = dataset_resolution
        self.dataset_indices = dataset_indices
        self.buffer = incremental_buffer
        self.max_iter = incremental_max_iter
        self.grad_max_iter = incremental_grad_max_iter

        if mg_patching_levels > 0:
            self.mg_n_patches = 2**mg_patching_levels
            if verbose:
                print(f'Training on {self.mg_n_patches**2} multi-grid patches.')
                sys.stdout.flush()
        else:
            self.mg_n_patches = 1
            mg_patching_padding = 0
            if verbose:
                print(f'Training on regular inputs (no multi-grid patching).')
                sys.stdout.flush()

        if self.incremental or self.incremental_resolution:
            self.incremental_scheduler = Incremental(model, incremental=self.incremental_grad, incremental_loss_gap=self.incremental_loss_gap,
                                                     incremental_resolution=self.incremental_resolution, incremental_eps = self.grad_eps, 
                                                     incremental_buffer=self.buffer, incremental_max_iter=self.max_iter, 
                                                     incremental_grad_max_iter=self.grad_max_iter, incremental_loss_eps = self.loss_eps,
                                                     incremental_res_gap = self.res_gap, dataset_resolution=self.dataset_resolution,
                                                     dataset_sublist=self.sublist, dataset_indices=self.dataset_indices)

        self.mg_patching_padding = mg_patching_padding
        self.patcher = MultigridPatching2D(model, levels=mg_patching_levels, padding_fraction=mg_patching_padding,
                                           use_distributed=use_distributed, stitching=mg_patching_stitching)

    def train(self, train_loader, test_loaders, output_encoder,
              model, optimizer, scheduler, regularizer, 
              training_loss=None, eval_losses=None):
        """Trains the given model on the given datasets"""
        n_train = len(train_loader.dataset)

        if not isinstance(test_loaders, dict):
            test_loaders = dict(test=test_loaders)

        if self.verbose:
            print(f'Training on {n_train} samples')
            print(f'Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples'
                  f'         on resolutions {[name for name in test_loaders]}.')
            sys.stdout.flush()

        if training_loss is None:
            training_loss = LpLoss(d=2)

        if eval_losses is None: # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)

        if output_encoder is not None:
            output_encoder.to(self.device)
        
        if self.use_distributed:
            is_logger = (comm.get_world_rank() == 0)
        else:
            is_logger = True 

        if self.incremental_loss_gap or self.incremental_grad:
            print("Model is initially using {} number of modes".format(model.fno_blocks.incremental_n_modes))

        for epoch in range(self.n_epochs):
            avg_loss = 0
            avg_lasso_loss = 0
            model.train()
            t1 = default_timer()
            train_err = 0.0

            for idx, sample in enumerate(train_loader):
                x, y = sample['x'], sample['y']
                
                if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f'Training on raw inputs of size {x.shape=}, {y.shape=}')

                x, y = self.patcher.patch(x, y)

                if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f'.. patched inputs of size {x.shape=}, {y.shape=}')

                x = x.to(self.device)
                y = y.to(self.device)

                # update the resolution
                if self.incremental_resolution:
                    x, y = self.incremental_scheduler.step(epoch=epoch, x=x, y=y)

                optimizer.zero_grad(set_to_none=True)
                if regularizer:
                    regularizer.reset()

                if self.amp_autocast:
                    with amp.autocast(enabled=True):
                        out = model(x)
                else:
                    out = model(x)
                if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f'Raw outputs of size {out.shape=}')

                out, y = self.patcher.unpatch(out, y)
                #Output encoding only works if output is stiched
                if output_encoder is not None and self.mg_patching_stitching:
                    out = output_encoder.decode(out)
                    y = output_encoder.decode(y)
                if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f'.. Processed (unpatched) outputs of size {out.shape=}')

                if self.amp_autocast:
                    with amp.autocast(enabled=True):
                        loss = training_loss(out.float(), y)
                else:
                    loss = training_loss(out.float(), y)

                if regularizer:
                    loss += regularizer.loss

                loss.backward()
                
                # update frequency modes based on the algorithm used
                if self.incremental:
                    self.incremental_scheduler.step(loss.item(), epoch)
                    
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

            train_err/= n_train
            avg_loss /= self.n_epochs
            
            if epoch % self.log_test_interval == 0: 
                
                msg = f'[{epoch}] time={epoch_train_time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}'

                values_to_log = dict(train_err=train_err, time=epoch_train_time, avg_loss=avg_loss)

                for loader_name, loader in test_loaders.items():
                    if epoch == self.n_epochs - 1 and self.log_output:
                        to_log_output = True
                    else:
                        to_log_output = False

                    errors = self.evaluate(model, eval_losses, loader, output_encoder, log_prefix=loader_name)

                    for loss_name, loss_value in errors.items():
                        msg += f', {loss_name}={loss_value:.4f}'
                        values_to_log[loss_name] = loss_value

                if regularizer:
                    avg_lasso_loss /= self.n_epochs
                    msg += f', avg_lasso={avg_lasso_loss:.5f}'

                if self.verbose and is_logger:
                    print(msg)
                    sys.stdout.flush()

                if self.incremental:
                    print("Model is currently using {} number of modes".format(model.fno_blocks.convs.incremental_n_modes))

                # Wandb loging
                if self.wandb_log and is_logger:
                    for pg in optimizer.param_groups:
                        lr = pg['lr']
                        values_to_log['lr'] = lr
                    wandb.log(values_to_log, step=epoch, commit=True)

    def evaluate(self, model, loss_dict, data_loader, output_encoder=None,
                 log_prefix=''):
        """Evaluates the model on a dictionary of losses
        
        Parameters
        ----------
        model : model to evaluate
        loss_dict : dict of functions 
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        output_encoder : used to decode outputs if not None
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary

        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """
        model.eval()

        if self.use_distributed:
            is_logger = (comm.get_world_rank() == 0)
        else:
            is_logger = True 

        errors = {f'{log_prefix}_{loss_name}':0 for loss_name in loss_dict.keys()}

        n_samples = 0
        with torch.no_grad():
            for it, sample in enumerate(data_loader):
                x, y = sample['x'], sample['y']

                n_samples += x.size(0)
                
                x, y = self.patcher.patch(x, y)
                y = y.to(self.device)
                x = x.to(self.device)

                if self.incremental_resolution:
                    x, y = self.incremental_scheduler.regularize_input_res(x, y)

                out = model(x)
        
                out, y = self.patcher.unpatch(out, y, evaluation=True)

                if output_encoder is not None:
                    out = output_encoder.decode(out)

                if (it == 0) and self.log_output and self.wandb_log and is_logger:
                    if out.ndim == 2:
                        img = out
                    else:
                        img = out.squeeze()[0]
                    wandb.log({f'image_{log_prefix}': wandb.Image(img.unsqueeze(-1).cpu().numpy())}, commit=False)
                
                for loss_name, loss in loss_dict.items():
                    errors[f'{log_prefix}_{loss_name}'] += loss(out, y).item()

        del x, y, out

        for key in errors.keys():
            errors[key] /= n_samples

        return errors

