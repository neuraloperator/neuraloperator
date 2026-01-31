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
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class AirfransTrainer(Trainer):



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

        if self.use_distributed and dist.is_initialized():
            device_id = dist.get_rank()
            self.model = DDP(self.model, device_ids=[device_id], output_device=device_id)

        if self.data_processor is not None:
            self.data_processor = self.data_processor.to(self.device)

        # ensure save_best is a metric we collect
        if self.save_best is not None:
            metrics = []
            for name in test_loaders.keys():
                for metric in eval_losses.keys():
                    metrics.append(f"{name}_{metric}")
            assert (
                self.save_best in metrics
            ), f"Error: expected a metric of the form <loader_name>_<metric>, got {save_best}"
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
                self.plot_diagnostic_grid(train_loader, epoch, save_dir=save_dir, sample_idx=sample_idx)

            # save checkpoint if save_every and save_best is not set
            if self.save_every is not None:
                if epoch % self.save_every == 0:
                    self.checkpoint(save_dir)

        return epoch_metrics

    
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

        #if self.data_processor is not None:
        #    out, sample = self.data_processor.postprocess(out, sample)

        loss = 0.0

        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                loss += training_loss(out, **sample)
        else:
            loss += training_loss(out, **sample)

        if self.regularizer:
            loss += self.regularizer.loss

        return loss

    
    def plot_diagnostic_grid(self, loader, epoch, save_dir="plots", sample_idx=0):
        """
        Diagnostic plot for 4 Input Channels & up to 4 Output Channels.
        Columns: [Raw In | Norm In] [Truth Norm | Pred Norm] [Truth Phys | Pred Phys | Residual]
        """
        self.model.eval()
        if self.data_processor:
            self.data_processor.eval()
        
        os.makedirs(save_dir, exist_ok=True)
        batch = next(iter(loader))
        
        with torch.no_grad():
            # 1. Process Data
            x_raw = batch['x'][sample_idx:sample_idx+1].to(self.device)
            y_raw = batch['y'][sample_idx:sample_idx+1].to(self.device)
            
            # Preprocess to get the Normalized signals the model actually sees
            sample = self.data_processor.preprocess({'x': x_raw, 'y': y_raw})
            x_norm = sample['x']
            y_norm_truth = sample['y']
            
            # Model Prediction
            y_norm_pred = self.model(x_norm)
            
            # Postprocess to get back to Physics
            y_phys_pred, _ = self.data_processor.postprocess(y_norm_pred.clone(), sample)

        # 2. Setup Plotting Grid
        # We plot Inputs in one section and Outputs in another
        n_in = x_raw.shape[1]
        n_out = y_raw.shape[1]
        
        fig = plt.figure(figsize=(25, 4 * max(n_in, n_out)))
        subfigs = fig.subfigures(2, 1, height_ratios=[n_in, n_out], hspace=0.1)
        
        # --- TOP: INPUT CHANNELS (Raw vs Normalized) ---
        in_axes = subfigs[0].subplots(n_in, 2)
        subfigs[0].suptitle(f"Epoch {epoch}: Input Signal Audit (SDF/Mask/Flow)", fontsize=16)
        
        for i in range(n_in):
            # Raw Input
            im0 = in_axes[i, 0].imshow(x_raw[0, i].cpu(), origin='lower')
            plt.colorbar(im0, ax=in_axes[i, 0])
            in_axes[i, 0].set_title(f"In Ch{i} Raw (Min: {x_raw[0,i].min():.2f})")
            
            # Normalized Input (The 'True' input to FNO)
            im1 = in_axes[i, 1].imshow(x_norm[0, i].cpu(), origin='lower', cmap='RdBu_r')
            plt.colorbar(im1, ax=in_axes[i, 1])
            in_axes[i, 1].set_title(f"In Ch{i} Norm (Min: {x_norm[0,i].min():.2f})")

        # --- BOTTOM: OUTPUT CHANNELS (Phys Truth, Norm Truth, Norm Pred, Phys Pred, Residual) ---
        out_axes = subfigs[1].subplots(n_out, 5)
        subfigs[1].suptitle(f"Epoch {epoch}: Output Field Audit (U/V/Cp)", fontsize=16)
        
        for i in range(n_out):
            # Truth Phys
            im2 = out_axes[i, 0].imshow(y_raw[0, i].cpu(), origin='lower')
            plt.colorbar(im2, ax=out_axes[i, 0])
            out_axes[i, 0].set_title(f"Truth Phys Ch{i}")

            # Truth Norm (Z-score)
            im3 = out_axes[i, 1].imshow(y_norm_truth[0, i].cpu(), origin='lower', cmap='plasma')
            plt.colorbar(im3, ax=out_axes[i, 1])
            out_axes[i, 1].set_title("Truth Norm (Z)")

            # Pred Norm (Model Output)
            im4 = out_axes[i, 2].imshow(y_norm_pred[0, i].cpu(), origin='lower', cmap='plasma')
            plt.colorbar(im4, ax=out_axes[i, 2])
            out_axes[i, 2].set_title("Pred Norm (Z)")

            # Phys Pred
            im5 = out_axes[i, 3].imshow(y_phys_pred[0, i].cpu(), origin='lower')
            plt.colorbar(im5, ax=out_axes[i, 3])
            out_axes[i, 3].set_title("Phys Pred")

            # Residual (Truth - Pred)
            res = y_raw[0, i] - y_phys_pred[0, i]
            im6 = out_axes[i, 4].imshow(res.cpu(), origin='lower', cmap='RdBu_r')
            plt.colorbar(im6, ax=out_axes[i, 4])
            out_axes[i, 4].set_title("Residual (Error)")

        plt.tight_layout()
        save_path = f"{save_dir}/diagnostic_epoch_{epoch:04d}.png"
        plt.savefig(save_path)
        
        if self.wandb_log:
            wandb.log({"diagnostic_grid": wandb.Image(save_path)}, step=epoch)
        
        plt.close(fig)