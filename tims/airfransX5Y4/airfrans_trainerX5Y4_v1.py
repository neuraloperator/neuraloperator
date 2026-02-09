from random import sample
from unittest import loader
import wandb
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.cuda import amp
from timeit import default_timer
from pathlib import Path
from typing import Union
import sys
import warnings
from neuralop.training.trainer import Trainer
from neuralop.losses import LpLoss
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tims.airfransX5Y4 import airfrans_utils as utils

class AirfransTrainer(Trainer):
    
    def __init__(self, *, model, n_epochs, wandb_log = False, device = "cpu", mixed_precision = False, data_processor = None, eval_interval = 1, log_output = False, use_distributed = False, verbose = False):
        self.utils = utils.AirfransUtils(model=model, data_processor=data_processor, device=device)
        super().__init__(model=model, n_epochs=n_epochs, wandb_log=wandb_log, device=device, mixed_precision=mixed_precision, data_processor=data_processor, eval_interval=eval_interval, log_output=log_output, use_distributed=use_distributed, verbose=verbose)

    def eval_one_batch(
        self, sample: dict, eval_losses: dict, return_output: bool = False
    ):
        """  Need to override original to ignore props in sample
            eval_one_batch runs inference on one batch
        and returns eval_losses for that batch.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_outputs : bool
            whether to return model outputs for plotting
            by default False
        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss name
        outputs: torch.Tensor | None
            optionally returns batch outputs
        """
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data to device if no preprocessor exists only tensor items
            sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}
        
        self.n_samples += sample["y"].size(0)

        out = self.model(**sample)

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)
        
        # Update this in your trainer
        loss_sample = {k: v for k, v in sample.items() if torch.is_tensor(v) or k == 'y'}

        eval_step_losses = {}

        for loss_name, loss_fn in eval_losses.items():
            # Extract y from loss_sample to pass as positional argument
            y_target = loss_sample.get('y', None)
            
            # Create kwargs without 'y' key
            loss_kwargs = {k: v for k, v in loss_sample.items() if k != 'y'}
            
            if y_target is not None:
                res = loss_fn(out, y_target, **loss_kwargs)  # pred, y, **kwargs
            #    print(f"Called loss_fn(out, y_target) -> result: {res}")
            else:
                res = loss_fn(out, **loss_sample)  # Fallback to original

            
            if isinstance(res, tuple):
                val_loss_out = res[0]
                print(f"Extracted loss from tuple: {val_loss_out}")
            else:
                val_loss_out = res

            eval_step_losses[loss_name] = val_loss_out.detach().item()

        if return_output:
            if isinstance(out, dict):
                out = out.get('y')

            return eval_step_losses, out
        else:
            return eval_step_losses, None


    def evaluate(self, loss_dict, data_loader, log_prefix="", mode="single_step", **kwargs):
        # Set model and data_processor to eval mode
        # to handle props in batch correctly
        # and return of tuple from loss functions
        
        self.model.eval()

        if self.data_processor:
            self.data_processor.eval()

        # Initialize as floats
        errors = {f"{log_prefix}_{loss_name}": 0.0 for loss_name in loss_dict.keys()}
        self.n_samples = 0
        local_n_samples =0

        with torch.no_grad():
            for idx, sample in enumerate(data_loader):
                # Use your overridden eval_one_batch which already pulls out the scalar
                eval_step_losses, outs = self.eval_one_batch(
                    sample, loss_dict, return_output=(idx == 0)
                )
                batch_size = sample['y'].size(0)
                local_n_samples += batch_size

                # self.n_samples is updated inside eval_one_batch based on y.size(0)
                for loss_name, val_loss in eval_step_losses.items():
                    # Ensure we are summing scalar floats
                    if torch.is_tensor(val_loss):
                        errors[f"{log_prefix}_{loss_name}"] += batch_size*val_loss.item()
                    else:
                        errors[f"{log_prefix}_{loss_name}"] += batch_size*val_loss

        # Normalize by n_samples (total number of airfoils seen)
        for key in errors.keys():
            if local_n_samples > 0:
                errors[key] /= local_n_samples
                print(f"Eval {key}: {errors[key]:.6f} over {local_n_samples} samples")
            else:
                print("Warning: local_n_samples is 0 during evaluation. Check data loader.")


        return errors


    def train(
        self,
        train_loader,
        test_loaders,
        optimizer,
        scheduler,
        regularizer=None,
        training_loss=None,
        eval_losses=None,
        eval_modes=None,
        save_every: int = None,
        save_best: int = None,
        save_dir: Union[str, Path] = "./ckpt",
        resume_from_dir: Union[str, Path] = None,
        max_autoregressive_steps: int = None,
        sample_idx: int = 0,
    ):
        """Trains the given model on the given dataset.

        If a device is provided, the model and data processor are loaded to device here.

        Parameters
        -----------
        train_loader: torch.utils.data.DataLoader
            training dataloader
        test_loaders: dict[torch.utils.data.DataLoader]
            testing dataloaders
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        scheduler: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: training.losses function
            cost function to minimize
        eval_losses: dict[Loss]
            dict of losses to use in self.eval()
        eval_modes: dict[str], optional
            optional mapping from the name of each loader to its evaluation mode.

            * if 'single_step', predicts one input-output pair and evaluates loss.

            * if 'autoregressive', autoregressively predicts output using last step's
            output as input for a number of steps defined by the temporal dimension of the batch.
            This requires specially batched data with a data processor whose ``.preprocess`` and
            ``.postprocess`` both take ``idx`` as an argument.
        save_every: int, optional, default is None
            if provided, interval at which to save checkpoints
        save_best: str, optional, default is None
            if provided, key of metric f"{loader_name}_{loss_name}"
            to monitor and save model with best eval result
            Overrides save_every and saves on eval_interval
        save_dir: str | Path, default "./ckpt"
            directory at which to save training states if
            save_every and/or save_best is provided
        resume_from_dir: str | Path, default None
            if provided, resumes training state (model, optimizer, regularizer, scheduler) 
            from state saved in `resume_from_dir`
        max_autoregressive_steps : int, default None
            if provided, and a dataloader is to be evaluated in autoregressive mode,
            limits the number of autoregressive in each rollout to be performed.

        Returns
        -------
        all_metrics: dict
            dictionary keyed f"{loader_name}_{loss_name}"
            of metric results for last validation epoch across
            all test_loaders

        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        if regularizer:
            self.regularizer = regularizer
        else:
            self.regularizer = None

        if training_loss is None:
            training_loss = LpLoss(d=2)

        # Warn the user if training loss is reducing across the batch
        if hasattr(training_loss, "reduction"):
            if training_loss.reduction == "mean":
                warnings.warn(
                    f"{training_loss.reduction=}. This means that the loss is "
                    "initialized to average across the batch dim. The Trainer "
                    "expects losses to sum across the batch dim."
                )

        if eval_losses is None:  # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)

        # accumulated wandb metrics
        self.wandb_epoch_metrics = None

        # create default eval modes
        if eval_modes is None:
            eval_modes = {}

        # attributes for checkpointing
        self.save_every = save_every
        self.save_best = save_best
        if resume_from_dir is not None:
            self.resume_state_from_dir(resume_from_dir)

        # Load model and data_processor to device
        self.model = self.model.to(self.device)

        if self.data_processor is not None:
            self.data_processor = self.data_processor.to(self.device)
            
            # Only save from the main process in DDP
            if not self.use_distributed or (dist.is_initialized() and dist.get_rank() == 0):
                # Ensure directory exists
                save_dir.mkdir(parents=True, exist_ok=True)

                self.save_dir = Path(save_dir)
                
                # Save to CPU to ensure ease of loading elsewhere
                torch.save(self.data_processor.state_dict(), self.save_dir / "data_processor.pt")
                
                if self.verbose:
                    print(f"✅ DataProcessor locked and saved to {save_dir}/data_processor.pt")
        else:
            if self.verbose:
                print("⚠️ No DataProcessor provided; ensure data is preprocessed appropriately.")

        # ensure save_best is a metric we collect
        if self.save_best is not None:
            metrics = []
            for name in test_loaders.keys():
                for metric in eval_losses.keys():
                    metrics.append(f"{name}_{metric}")

            if self.save_best not in metrics:
                    valid_keys = list(metrics)
                    # Filter out non-scalar values or metadata if necessary
                    error_msg = (
                        f"\n" + "!"*50 + "\n"
                        f"Error: 'save_best' metric '{self.save_best}' not found in logs.\n"
                        f"Available metrics in this run:\n"
                        f"  - " + "\n  - ".join(valid_keys) + "\n"
                        f"Hint: Ensure the format is '<loader_name>_<loss_name>'.\n"
                        + "!"*50
                    )
                    raise AssertionError(error_msg)

            assert (
                self.save_best in metrics
            ), error_msg
            best_metric_value = float("inf")
            # either monitor metric or save on interval, exclusive for simplicity
            self.save_every = None

        if self.verbose:
            print(f"Training on {len(train_loader.dataset)} samples")
            print(f"Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples"
                f"         on resolutions {[name for name in test_loaders]}.")
            sys.stdout.flush()
        
        

        for epoch in range(self.start_epoch, self.n_epochs):
            (
                train_err,
                avg_loss,
                avg_lasso_loss,
                epoch_train_time,
            ) = self.train_one_epoch(epoch, train_loader, training_loss)
            epoch_metrics = dict(
                train_err=train_err,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                epoch_train_time=epoch_train_time,
            )

            if epoch % self.eval_interval == 0:
                # evaluate and gather metrics across each loader in test_loaders
                eval_metrics = self.evaluate_all(
                    epoch=epoch,
                    eval_losses=eval_losses,
                    test_loaders=test_loaders,
                    eval_modes=eval_modes,
                    max_autoregressive_steps=max_autoregressive_steps,
                )
                epoch_metrics.update(**eval_metrics)
                # save checkpoint if conditions are met
                if save_best is not None:
                    if eval_metrics[save_best] < best_metric_value:
                        best_metric_value = eval_metrics[save_best]
                        self.checkpoint(save_dir)
                
                #plot diagnostic grid every eval_interval
                self.plot_diagnostic_grid(train_loader, epoch, save_dir=save_dir, sample_idx=sample_idx,training_loss=training_loss)

            # save checkpoint if save_every and save_best is not set
            if self.save_every is not None:
                if epoch % self.save_every == 0:
                    self.checkpoint(save_dir)

        return epoch_metrics

    def train_one_epoch(self, epoch, train_loader, training_loss):
        """train_one_epoch trains self.model on train_loader
        for one epoch and returns training metrics

        Parameters
        ----------
        epoch : int
            epoch number
        train_loader : torch.utils.data.DataLoader
            data loader of train examples
        test_loaders : dict
            dict of test torch.utils.data.DataLoader objects

        Returns
        -------
        all_errors
            dict of all eval metrics for the last epoch
        """
        self.on_epoch_start(epoch)
        avg_loss = 0
        avg_lasso_loss = 0
        self.model.train()
        if self.data_processor:
            self.data_processor.train()
        t1 = default_timer()
        train_err = 0.0
        avg_channel_loss = torch.zeros(self.model.out_channels).to(self.device)
        avg_channel_err = torch.zeros(self.model.out_channels).to(self.device)
        # track number of training examples in batch
        self.n_samples = 0


        # number of batches
        n_batches = len(train_loader)

        for idx, sample in enumerate(train_loader):

            current_props = sample.get('props', None)

            loss , loss_per_channel = self.train_one_batch(idx, sample, training_loss)
            loss.backward()
            self.optimizer.step()

            train_err += loss.item()
            with torch.no_grad():
                avg_loss += loss.item()
                if self.regularizer:
                    avg_lasso_loss += self.regularizer.loss

                # --- FIXED ACCUMULATION ---
                if isinstance(loss_per_channel, dict):
                    # Map the dictionary keys back to the indices for your pre-allocated tensor
                    # Match these strings exactly to what your MetaLoss returns
                    avg_channel_loss[0] += loss_per_channel.get('enc_u_loss', 0.0)
                    avg_channel_loss[1] += loss_per_channel.get('enc_v_loss', 0.0)
                    avg_channel_loss[2] += loss_per_channel.get('enc_cp_loss', 0.0)
                    avg_channel_loss[3] += loss_per_channel.get('enc_lognutratio_loss', 0.0)
                # Fallback for standard list/tensor return types
                elif isinstance(loss_per_channel, (list, torch.Tensor)):
                    for c, c_loss in enumerate(loss_per_channel):
                        if c < avg_channel_loss.size(0):
                            avg_channel_loss[c] += c_loss.item()
                

                # 2. Diagnostic Block (Check interval and ensure it's the first batch)
                if (epoch % self.eval_interval == 0) and (idx ==0):
                    self.model.eval() # Set to eval mode for consistent interpolation
                    with torch.no_grad():
                        # Move data to device and get physical predictions
                        x = sample['x'].to(self.device)
                        y_pred = self.model(x)
                        
                        # temporarily remove 'props' to avoid issues in postprocess
                        #props_meta = sample.pop('props', None)
                        # filter out only tensor items for postprocess
                        tensor_sample = {k: v for k, v in sample.items() if isinstance(v, torch.Tensor)}
                        
                        # Get decoded physical values for plotting
                        y_decoded, _ = self.data_processor.postprocess(y_pred, tensor_sample)
                        y_phys = y_decoded['y']
                        
                        
                        sample['props'] = current_props


                        # Run the CL predictions on a batch
                        cl_preds = self.utils.evaluate_batch_metrics(sample)
                        
                        # Plot the Cp distribution for the first airfoil in this batch
                        clp_truth, clp_pred = self.utils.plot_surface_diagnostic(
                            y_phys=y_phys, 
                            batch=sample, 
                            sample_idx=0,
                            save_dir=self.save_dir
                        )
                        print(f" >> Sampled CL Predictions: Truth {clp_truth:.4f}  Predicted {clp_pred:.4f} ")

                    
                    self.model.train() # Switch back to training mode
    

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_err)
        else:
            self.scheduler.step()

        epoch_train_time = default_timer() - t1

        train_err /= n_batches
        avg_loss /= self.n_samples

        avg_channel_loss /= n_batches

        # Store in dict for logging
        # .item() converts 0-dim tensor to python float
        channel_metrics = {
            "enc_u_loss": avg_channel_loss[0].item(), 
            "enc_v_loss": avg_channel_loss[1].item(), 
            "enc_cp_loss": avg_channel_loss[2].item(), 
            "enc_lognutratio_loss": avg_channel_loss[3].item()
        }
        if self.regularizer:
            avg_lasso_loss /= self.n_samples
        else:
            avg_lasso_loss = None

        lr = None
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
        if self.verbose and epoch % self.eval_interval == 0:
            self.log_training(
                epoch=epoch,
                time=epoch_train_time,
                avg_loss=avg_loss,
                train_err=train_err,
                channel_metrics=channel_metrics,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
            )

        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

    
    def train_one_batch(self, idx, sample, training_loss):
        """Run one batch of input through model
           and return training loss on outputs
           modified to calculate loss on normalized outputs

        Parameters
        ----------
        idx : int
            index of batch within train_loader
        sample : dict
            data dictionary holding one batch

        Returns
        -------
        loss: float | Tensor
            float value of training loss
        """

        self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer:
            self.regularizer.reset()
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data to device if no preprocessor exists
            sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}

        if isinstance(sample["y"], torch.Tensor):
            self.n_samples += sample["y"].shape[0]
        else:
            self.n_samples += 1

        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                out = self.model(**sample)
        else:
            out = self.model(**sample)
        
        if self.epoch == 0 and idx == 0 and self.verbose and isinstance(out, torch.Tensor):
            print(f"Raw outputs of shape {out.shape}")

        # loss calculation on encoded outputs commented out postprocessing
        #if self.data_processor is not None:
        #    out, sample = self.data_processor.postprocess(out, sample)

        loss = 0.0

        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                # weighted losses return (loss, weights),which is a tuple
                loss_out = training_loss(out, **sample)
                if isinstance(loss_out, tuple):
                    batch_loss, channel_batch_losses = loss_out
                else:   
                    batch_loss = loss_out
                loss += batch_loss
        else:
            loss_out = training_loss(out, **sample)
            # if weighted losses return (loss, weights),which is a tuple
            if isinstance(loss_out, tuple):
                batch_loss, channel_batch_losses = loss_out
            else:   
                batch_loss = loss_out
            loss += batch_loss

        if self.regularizer:
            loss += self.regularizer.loss



        return loss, channel_batch_losses
    
        
    def log_training(
        self,
        epoch: int,
        time: float,
        avg_loss: float,
        train_err: float,
        channel_metrics: dict = None,
        avg_lasso_loss: float = None,
        lr: float = None,
    ):
        """Extended method to log results
        from a single training epoch.


        Parameters
        ----------
        epoch: int
        time: float
            training time of epoch
        avg_loss: float
            average train_err per individual sample
        train_err: float
            train error for entire epoch
        channel_metrics: dict
            per-channel metrics to log
        avg_lasso_loss: float
            average lasso loss from regularizer, optional
        lr: float
            learning rate at current epoch
        """
        # accumulate info to log to wandb
        if self.log_output:
            if self.wandb_log:
                values_to_log = dict(
                    train_err=train_err,
                    time=time,
                    avg_loss=avg_loss,
                    avg_lasso_loss=avg_lasso_loss,
                    lr=lr,
                    **(channel_metrics or {})
                )

            #msg = f"[{epoch}] time={time:.2f}, "
            #msg += f"avg_loss={avg_loss:.4f}, "
            #msg += f"train_err={train_err:.4f},"
            # Only try to append channel metrics if they exist
            #if channel_metrics:
            #    # for key, value in channel_metrics.items():
            #    for key, value in channel_metrics.items():
            #        msg += f", {key}={value:.4f}"  

            
            if avg_lasso_loss is not None:
                msg += f", lasso={avg_lasso_loss:.4f}"
            if avg_lasso_loss is not None:
                msg += f", avg_lasso={avg_lasso_loss:.4f}"

            print(msg)
            sys.stdout.flush()

            if self.wandb_log:
                wandb.log(data=values_to_log, step=epoch + 1, commit=False)
            
            # --- LOCAL CSV LOGGING ---
            #if self.log_output: # Only Rank 0 writes to the file
            log_path = self.save_dir / "loss_history.csv"


            # 1. Create header if file is new
            if not log_path.exists():
                with open(log_path, 'w') as f:
                    # Customize this header to match your channel names
                    f.write("epoch, time, train_err, enc_u_loss, enc_v_loss, enc_cp_loss, enc_lognutratio_loss, lr\n")
            # 2. Extract values safely (decoupled from self.regularizer)
            # We use .get() to avoid KeyErrors if a channel is missing
            u = channel_metrics.get('enc_u_loss', 0.0) if channel_metrics else 0.0
            v = channel_metrics.get('enc_v_loss', 0.0) if channel_metrics else 0.0
            cp = channel_metrics.get('enc_cp_loss', 0.0) if channel_metrics else 0.0
            nut = channel_metrics.get('enc_lognutratio_loss', 0.0) if channel_metrics else 0.0
            
            # 3. Always log if we have data
            with open(log_path, 'a') as f:
                f.write(f"{epoch},{time:.2f},{train_err:.6f},{u:.6f},{v:.6f},{cp:.6f},{nut:.6f},{lr:.2e}\n")

            # 4. Standard Console Output
            msg = f"[{epoch}] time={time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}, "
            msg += f"enc_u_loss={u:.4f}, enc_v_loss={v:.4f}, enc_cp_loss={cp:.4f}, enc_lognutratio_loss={nut:.4f}"
            print(msg)
        if self.regularizer:          
            # 2. Append the current epoch's data
            with open(log_path, 'a') as f:              
                f.write(f"{epoch}, {time:.2f}, {train_err:.6f}, {u:.6f}, {v:.6f}, {cp:.6f}, {nut:.6f}, {lr:.2e}\n")
        else:
            print(f"Logging disabled on  {self.device} not logging  metrics. ")


    
    def plot_diagnostic_grid(self, loader, epoch, save_dir="plots", sample_idx=0, prefix="prediction",training_loss=None):
        """
        Diagnostic plot for 4 Input Channels & up to 4 Output Channels.
        Columns: [Raw In | Norm In] [Truth Norm | Pred Norm] [Truth Phys | Pred Phys | Residual]
        """
        self.model.eval()
        if self.data_processor:
            self.data_processor.eval()
        
        batch = next(iter(loader))

        #expecting aggregated fieldwise loss with target losses and weights for each channel
        target_losses = training_loss.losses
        target_weights = training_loss.weights      

        with torch.no_grad():
            x_raw = batch['x'][sample_idx:sample_idx+1].to(self.device)
            y_raw = batch['y'][sample_idx:sample_idx+1].to(self.device)
            # Convert to encoded space
            sample = self.data_processor.preprocess({'x': x_raw, 'y': y_raw})
            x_input = sample['x'].to(self.device)
            y_norm_truth = sample['y']
            y_norm_pred = self.model(x_input)

            # Trainer uses normalized losses so calculate normalized residuals
            residual_norm = y_norm_truth - y_norm_pred
            # For visualization, also calculate raw residuals
            y_decoded_dict,_ = self.data_processor.postprocess(y_norm_pred.clone(), sample)
            # --- THE FIX ---
            # If it's a dict, extract the 'y' tensor so the [:, i:i+1, ...] slice works
            if isinstance(y_decoded_dict, dict):
                y_decoded_pred = y_decoded_dict['y']
            else:
                y_decoded_pred = y_decoded_dict

            # Calc losses per channel
            n_channels = y_norm_truth.shape[1]

            encoded_channel_losses=[]
            encoded_weighted_losses =[]
            decoded_channel_losses=[]
            encoded_mse_losses =[]
            for i, field_name  in enumerate(target_losses):

                loss_fn = target_losses.get(field_name, LpLoss(d=2, p=2, reduction='mean'))
                weight = target_weights.get(field_name, 1.0)
                channel_residual = residual_norm[:, i:i+1, ...]
                encoded_channel_loss = loss_fn(y_norm_pred[:, i:i+1, ...], y_norm_truth[:, i:i+1, ...])
                encoded_channel_losses.append(encoded_channel_loss)
                encoded_mse_loss = torch.mean(channel_residual**2).item()
                encoded_mse_losses.append(encoded_mse_loss)
                encoded_channel_loss = weight * encoded_channel_loss
                encoded_weighted_losses.append(encoded_channel_loss)
                # Decoded loss
                decoded_channel_loss = loss_fn(
                    y_decoded_pred[:, i:i+1, ...],
                    y_raw[:, i:i+1, ...]
                )
                decoded_channel_losses.append(decoded_channel_loss)
                print(f"Epoch {epoch} Sample {sample_idx} Channel {field_name} - Loss: {encoded_channel_loss.item():.6f} Weighted Loss: {encoded_channel_loss.item():.6f}  MSE {encoded_mse_loss:.6f}   Decoded Loss: {decoded_channel_loss.item():.6f}")

            
            # Physics Pred
            residual = y_raw - y_decoded_pred


            # get resolution info
            resolution_h = x_raw.shape[-2] # 128
            resolution_w = x_raw.shape[-1] # 128

        n_out = y_raw.shape[1]
        fig, axes = plt.subplots(n_out, 5, figsize=(25, 4 * n_out))
        fig.suptitle(f"Airfrans Prediction: U-def, V-def, Cp , log(nut/nu) : Epoch {epoch}: Sample {sample_idx}", fontsize=20)

        n_labels = ['U-def', 'V-def', 'Cp', 'log(nut/nu)']

        residual_cmap_ranges = [0.01, 0.01, 0.01, 4.0]

        for i in range(n_out):
            # --- 1. Extract Scaling Bounds from Decoded Output ---
            # Physics Bounds
            cp_min, cp_max = y_raw[0, i].min().item(), y_raw[0, i].max().item()
            # Normalized (Z) Bounds
            z_min, z_max = y_norm_truth[0, i].min().item(), y_norm_truth[0, i].max().item()
            
            # --- 2. Calculate Stats ---
            res = residual[0, i].cpu().numpy()
            stats_text = (f"Decoded Range: [{cp_min:.1f}, {cp_max:.1f}]\n"
                        f"MAE: {np.abs(res).mean():.4f}")

            # Column 1: Truth  
            im0 = axes[i, 0].imshow(y_raw[0, i].cpu(), origin='lower', vmin=cp_min, vmax=cp_max)
            axes[i, 0].set_title(f"Truth {n_labels[i]} \n{stats_text}", fontsize=10, loc='left')
            plt.colorbar(im0, ax=axes[i, 0])

            # Column 2: Truth Encoded 
            im1 = axes[i, 1].imshow(y_norm_truth[0, i].cpu(), origin='lower', cmap='plasma', vmin=z_min, vmax=z_max)
            axes[i, 1].set_title(f"{n_labels[i]} Encoded \nRange: [{z_min:.2f}, {z_max:.2f}]")
            plt.colorbar(im1, ax=axes[i, 1])

            # Column 3: Pred Encoded 
            im2 = axes[i, 2].imshow(y_norm_pred[0, i].cpu(), origin='lower', cmap='plasma', vmin=z_min, vmax=z_max)
            axes[i, 2].set_title(f"Pred {n_labels[i]} Encoded  Residual {encoded_channel_losses[i].item():.6f} MSE {encoded_mse_losses[i]:.6f}\n")
            plt.colorbar(im2, ax=axes[i, 2])

            # Column 4: Prediction Decoded 
            im3 = axes[i, 3].imshow(y_decoded_pred[0, i].cpu(), origin='lower', vmin=cp_min, vmax=cp_max)
            axes[i, 3].set_title(f"Pred  {n_labels[i]}  Residual {decoded_channel_losses[i].item():.6f} \n")
            plt.colorbar(im3, ax=axes[i, 3])

            # Column 5: Residual Error (Dynamic scale for error detail)
            # Residuals are better with dynamic scale to see where the error 'lives'
            # fixed range per channel for consistency across plots
            max_err = np.max(np.abs(residual_cmap_ranges[i]))
            im4 = axes[i, 4].imshow(res, origin='lower', cmap='RdBu_r', vmin=-max_err, vmax=max_err)
            #l2loss = np.square(res).sum()
            axes[i, 4].set_title(f" Residual {decoded_channel_losses[i].item():.6f}")
            plt.colorbar(im4, ax=axes[i, 4])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        outpufile_dir = Path(f"{save_dir}/field")
        outpufile_dir.mkdir(parents=True, exist_ok=True)    
        plt.savefig(f"{outpufile_dir}/{prefix}_{resolution_h}x{resolution_w}_sample_{sample_idx}_epoch_{epoch:04d}.png")
        plt.close()
    
