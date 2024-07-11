import torch
from torch.cuda import amp
from timeit import default_timer
import pathlib
import sys
import wandb

from .callbacks import PipelineCallback
import neuralop.mpu.comm as comm
from neuralop.losses import LpLoss


class Trainer:
    def __init__(
        self,
        *,
        model,
        n_epochs,
        wandb_log=False,
        device=None,
        amp_autocast=False,
        data_processor=None,
        callbacks=None,
        log_test_interval=1,
        log_output=False,
        use_distributed=False,
        verbose=False,
    ):
        """
        A general Trainer class to train neural-operators on given datasets

        Parameters
        ----------
        model : nn.Module
        n_epochs : int
        wandb_log : bool, default is False
            whether to log results to wandb
        device : str 'cpu' or 'cuda'
        amp_autocast : bool, default is False
            whether to use torch.amp automatic mixed precision
        data_processor : DataProcessor class to transform data, default is None
            if not None, data from the loaders is transform first with data_processor.preprocess,
            then after getting an output from the model, that is transformed with data_processor.postprocess.
        log_test_interval : int, default is 1
            how frequently to print updates
        log_output : bool, default is False
            if True, and if wandb_log is also True, log output images to wandb
        use_distributed : bool, default is False
            whether to use DDP
        verbose : bool, default is False
        """

        self.model = model
        self.n_epochs = n_epochs

        self.wandb_log = wandb_log
        self.log_test_interval = log_test_interval
        self.log_output = log_output
        self.verbose = verbose
        self.use_distributed = use_distributed
        self.device = device
        self.amp_autocast = amp_autocast
        self.data_processor = data_processor

    def train(
        self,
        train_loader,
        test_loaders,
        optimizer,
        scheduler,
        regularizer,
        training_loss=None,
        eval_losses=None,
    ):
        """Trains the given model on the given datasets.
        params:
        train_loader: torch.utils.data.DataLoader
            training dataloader
        test_loaders: dict[torch.utils.data.DataLoader]
            testing dataloaders
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        optimizer: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: training.losses function
            cost function to minimize
        eval_losses: dict[Loss]
            dict of losses to use in self.eval()
        """

        # setup self.training_loss and self.eval_losses
        self.before_training_loop(
            train_loader=train_loader,
            test_loaders=test_loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            regularizer=regularizer,
            training_loss=training_loss,
            eval_losses=eval_losses,
            data_processor=self.data_processor,
            )
        
        errors = None

        for epoch in range(self.n_epochs):
            self.on_epoch_start(epoch)

            avg_loss = 0
            avg_lasso_loss = 0
            self.model.train()
            if self.data_processor:
                self.data_processor.train()
            t1 = default_timer()
            train_err = 0.0
            
            # track number of training examples in batch
            self.n_samples = 0

            for idx, sample in enumerate(train_loader):
                
                loss = self.train_one_batch(idx, sample)
                train_err += loss.item()

                loss.backward()
                #del out
                self.optimizer.step()

                with torch.no_grad():
                    avg_loss += loss.item()
                    if self.regularizer:
                        avg_lasso_loss += self.regularizer.loss

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(train_err)
            else:
                self.scheduler.step()

            epoch_train_time = default_timer() - t1

            train_err /= len(train_loader)
            avg_loss /= self.n_samples
            if regularizer:
                avg_lasso_loss /= self.n_samples
            else:
                avg_lasso_loss = None

            # collect info to log, message to print
            if epoch % self.log_test_interval == 0:
                
                all_errors = {}
                for loader_name, loader in test_loaders.items():
                    errors = self.evaluate(eval_losses, loader,
                                           log_prefix=loader_name)                        
                    all_errors.update(**errors)

                # print msg to console and optionally log to wandb
                if self.verbose:
                    lr = None
                    for pg in optimizer.param_groups:
                        lr = pg["lr"]
                    self.log_epoch(
                        epoch=epoch,
                        time=epoch_train_time,
                        avg_loss=avg_loss,
                        train_err=train_err,
                        avg_lasso_loss=avg_lasso_loss,
                        eval_metrics=all_errors,
                        lr=lr
                    )

        return all_errors

    def evaluate(self, loss_dict, data_loader, log_prefix=""):
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

        self.model.eval()
        if self.data_processor:
            self.data_processor.eval()

        errors = {f"{log_prefix}_{loss_name}": 0 for loss_name in loss_dict.keys()}

        self.n_samples = 0
        with torch.no_grad():
            for idx, sample in enumerate(data_loader):
                eval_step_losses = self.eval_one_batch(idx, sample)

                for loss_name, val_loss in eval_step_losses.items():
                    errors[f"{log_prefix}_{loss_name}"] += val_loss

        for key in errors.keys():
            errors[key] /= self.n_samples
            
        #del out

        return errors
    
    def before_training_loop(self,
                             train_loader,
                             test_loaders,
                             optimizer,
                             scheduler,
                             regularizer,
                             training_loss,
                             eval_losses,
                             data_processor):
        if training_loss is None:
            self.training_loss = LpLoss(d=2)
        else:
            self.training_loss = training_loss

        if eval_losses is None:  # By default just evaluate on the training loss
            self.eval_losses = dict(l2=training_loss)
        else:
            self.eval_losses = eval_losses
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.regularizer = regularizer
        
        if self.verbose:
            print(f'Training on {len(train_loader)} samples')
            print(f'Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples'
                  f'         on resolutions {[name for name in test_loaders]}.')
            sys.stdout.flush()
    
    def on_epoch_start(self, epoch):
        self.epoch = epoch
        return None

    def train_one_batch(self, idx, sample):

        self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer:
            self.regularizer.reset()

        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data to device if no preprocessor exists
            sample = {
                k: v.to(self.device)
                for k, v in sample.items()
                if torch.is_tensor(v)
            }

        self.n_samples += sample["y"].shape[0]

        if self.amp_autocast:
            with amp.autocast(enabled=True):
                out = self.model(**sample)
        else:
            out = self.model(**sample)
        
        if self.epoch == 0 and idx == 0 and self.verbose:
            print(f"Raw outputs of shape {out.shape}")

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        loss = 0.0

        if self.amp_autocast:
            with amp.autocast(enabled=True):
                loss += self.training_loss(out, **sample)
        else:
            loss += self.training_loss(out, **sample)

        if self.regularizer:
            loss += self.regularizer.loss

        return loss
    
    def eval_one_batch(self, idx, sample):
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)

        self.n_samples += sample["y"].size(0)

        out = self.model(**sample)

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)
        
        eval_step_losses = {}

        for loss_name, loss in self.eval_losses.items():
            val_loss = loss(out, **sample)
            if val_loss.shape == ():
                val_loss = val_loss.item()
            eval_step_losses[loss_name] = val_loss
        return eval_step_losses
        
            

    def log_epoch(self, 
            epoch:int,
            time: float,
            avg_loss: float,
            train_err: float,
            avg_lasso_loss: float=None,
            eval_metrics: dict=None,
            lr: float=None):
        """Basic method to log a dict of output values
        from a single training epoch. 
        

        Parameters
        ----------
        values : dict
            dict keyed 'metric': float_value
        """
        if self.wandb_log:
            values_to_log = dict(
                train_err=train_err,
                time=time,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr)

        msg = f"[{epoch}] time={time:.2f}, "
        msg += f"avg_loss={avg_loss:.4f}, "
        msg += f"train_err={train_err:.4f}"
        if avg_lasso_loss is not None:
            msg += f", avg_lasso={avg_lasso_loss:.4f}"
        if eval_metrics:
            for metric, value in eval_metrics.items():
                msg += f", {metric}={value:.4f}"
                if self.wandb_log:
                    values_to_log[metric] = value

        print(msg)
        sys.stdout.flush()

        if self.wandb_log and wandb.run is not None:
            wandb.log(
                values_to_log,
                step=epoch+1,
                commit=True
            )
