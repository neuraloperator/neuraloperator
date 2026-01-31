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

class AirfransVelCpTrainer(Trainer):

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
        
        batch = next(iter(loader))
        with torch.no_grad():
            x_raw = batch['x'][sample_idx:sample_idx+1].to(self.device)
            y_raw = batch['y'][sample_idx:sample_idx+1].to(self.device)
            
            sample = self.data_processor.preprocess({'x': x_raw, 'y': y_raw})
            y_norm_truth = sample['y']
            y_norm_pred = self.model(sample['x'])
            
            # Physics Pred
            y_phys_pred, _ = self.data_processor.postprocess(y_norm_pred.clone(), sample)
            residual = y_raw - y_phys_pred

            # get resolution info
            resolution_h = x_raw.shape[-2] # 128
            resolution_w = x_raw.shape[-1] # 128

        n_out = y_raw.shape[1]
        fig, axes = plt.subplots(n_out, 5, figsize=(25, 4 * n_out))
        fig.suptitle(f"Airfrans Prediction: U, V, Cp : Epoch {epoch}: Sample {sample_idx}", fontsize=20)

        n_labels = ['U', 'V', 'Cp', 'nut']

        for i in range(n_out):
            # --- 1. Extract Scaling Bounds from Truth ---
            # Physics Bounds
            p_min, p_max = y_raw[0, i].min().item(), y_raw[0, i].max().item()
            # Normalized (Z) Bounds
            z_min, z_max = y_norm_truth[0, i].min().item(), y_norm_truth[0, i].max().item()
            
            # --- 2. Calculate Stats ---
            res = residual[0, i].cpu().numpy()
            stats_text = (f"Truth Range: [{p_min:.1f}, {p_max:.1f}]\n"
                        f"MAE: {np.abs(res).mean():.4f}")

            # Column 1: Target Phys 
            im0 = axes[i, 0].imshow(y_raw[0, i].cpu(), origin='lower', vmin=p_min, vmax=p_max)
            axes[i, 0].set_title(f"Target {n_labels[i]} \n{stats_text}", fontsize=10, loc='left')
            plt.colorbar(im0, ax=axes[i, 0])

            # Column 2: Truth Norm 
            im1 = axes[i, 1].imshow(y_norm_truth[0, i].cpu(), origin='lower', cmap='plasma', vmin=z_min, vmax=z_max)
            axes[i, 1].set_title(f"{n_labels[i]} Norm (Z)\nRange: [{z_min:.2f}, {z_max:.2f}]")
            plt.colorbar(im1, ax=axes[i, 1])

            # Column 3: Pred Norm 
            im2 = axes[i, 2].imshow(y_norm_pred[0, i].cpu(), origin='lower', cmap='plasma', vmin=z_min, vmax=z_max)
            axes[i, 2].set_title(f"Pred {n_labels[i]} Norm (Z)\n")
            plt.colorbar(im2, ax=axes[i, 2])

            # Column 4: Physics Pred 
            im3 = axes[i, 3].imshow(y_phys_pred[0, i].cpu(), origin='lower', vmin=p_min, vmax=p_max)
            axes[i, 3].set_title(f"Prediction {n_labels[i]} \n")
            plt.colorbar(im3, ax=axes[i, 3])

            # Column 5: Residual Error (Dynamic scale for error detail)
            # Residuals are better with dynamic scale to see where the error 'lives'
            im4 = axes[i, 4].imshow(res, origin='lower', cmap='RdBu_r')

            l2loss = np.square(res).sum()
            axes[i, 4].set_title(f"Residual sum(L2): {l2loss:.4f}")
            plt.colorbar(im4, ax=axes[i, 4])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{save_dir}/prediction_{resolution_h}x{resolution_w}_sample_{sample_idx}_epoch_{epoch:04d}.png")
        plt.close()