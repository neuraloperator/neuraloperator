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

from .training_state import save_training_state


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
        overrides_loss = [
            "compute_training_loss" in c.__class__.__dict__.keys() for c in callbacks
        ]

        if sum(overrides_loss) >= 1:
            self.overrides_loss = True
            print("using custom callback to compute loss.")
        else:
            self.overrides_loss = False
            print("using standard method to compute loss.")

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
        if self.overrides_loss:
            for c in self.callbacks:
                c.compute_training_loss(*args, **kwargs)
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
        if self.overrides_loss:
            for c in self.callbacks:
                c.compute_val_loss(*args, **kwargs)
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


class BasicLoggerCallback(Callback):
    """
    Callback that implements simple logging functionality
    expected when passing verbose to a Trainer
    """

    def __init__(self, wandb_kwargs=None):
        super().__init__()
        if wandb_kwargs:
            wandb.init(**wandb_kwargs)

    def on_init_end(self, *args, **kwargs):
        self._update_state_dict(**kwargs)

    def on_train_start(self, **kwargs):
        self._update_state_dict(**kwargs)

        train_loader = self.state_dict["train_loader"]
        test_loaders = self.state_dict["test_loaders"]
        verbose = self.state_dict["verbose"]

        n_train = len(train_loader.dataset)
        self._update_state_dict(n_train=n_train)

        if not isinstance(test_loaders, dict):
            test_loaders = dict(test=test_loaders)

        if verbose:
            print(f"Training on {n_train} samples")
            print(
                f"Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples"
                f"         on resolutions {[name for name in test_loaders]}."
            )
            sys.stdout.flush()

    def on_epoch_start(self, epoch):
        self._update_state_dict(epoch=epoch)

    def on_batch_start(self, idx, **kwargs):
        self._update_state_dict(idx=idx)

    def on_before_loss(self, out, **kwargs):
        if (
            self.state_dict["epoch"] == 0
            and self.state_dict["idx"] == 0
            and self.state_dict["verbose"]
        ):
            print(f"Raw outputs of size {out.shape=}")

    def on_before_val(self, epoch, train_err, time, avg_loss, avg_lasso_loss, **kwargs):
        # track training err and val losses to print at interval epochs
        msg = f"[{epoch}] time={time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}"
        values_to_log = dict(train_err=train_err, time=time, avg_loss=avg_loss)

        self._update_state_dict(msg=msg, values_to_log=values_to_log)
        self._update_state_dict(avg_lasso_loss=avg_lasso_loss)

    def on_val_epoch_end(self, errors, **kwargs):
        for loss_name, loss_value in errors.items():
            if isinstance(loss_value, float):
                self.state_dict["msg"] += f", {loss_name}={loss_value:.4f}"
            else:
                loss_value = {i: e.item() for (i, e) in enumerate(loss_value)}
                self.state_dict["msg"] += f", {loss_name}={loss_value}"
            self.state_dict["values_to_log"][loss_name] = loss_value

    def on_val_end(self, *args, **kwargs):
        if self.state_dict.get("regularizer", False):
            avg_lasso = self.state_dict.get("avg_lasso_loss", 0.0)
            avg_lasso /= self.state_dict.get("n_epochs")
            self.state_dict["msg"] += f", avg_lasso={avg_lasso:.5f}"

        print(self.state_dict["msg"])
        sys.stdout.flush()

        if self.state_dict.get("wandb_log", False):
            for pg in self.state_dict["optimizer"].param_groups:
                lr = pg["lr"]
                self.state_dict["values_to_log"]["lr"] = lr
            wandb.log(
                self.state_dict["values_to_log"],
                step=self.state_dict["epoch"] + 1,
                commit=True,
            )


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

    def on_init_end(self, *args, **kwargs):
        self._update_state_dict(**kwargs)

    def on_train_start(self, *args, **kwargs):
        self._update_state_dict(**kwargs)

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
            saved_modules = [x.stem for x in self.resume_from_dir.glob("*.pt")]

            assert (
                "best_model_state_dict" in saved_modules
                or "model_state_dict" in saved_modules
            ), "Error: CheckpointCallback expects a model state dict named model.pt or best_model.pt."

            # no need to handle exceptions if assertion that either model file exists passes
            if "best_model_state_dict" in saved_modules:
                if hasattr(self.state_dict["model"], "load_checkpoint"):
                    self.state_dict["model"].load_checkpoint(
                        save_folder=self.resume_from_dir, save_name="best_model"
                    )
                else:
                    self.state_dict["model"].load_state_dict(
                        torch.load(self.resume_from_dir / "best_model.pt")
                    )
                if verbose:
                    print(f"Loading model state from best_model_state_dict.pt")
            else:
                if hasattr(self.state_dict["model"], "load_checkpoint"):
                    self.state_dict["model"].load_checkpoint(
                        save_folder=self.resume_from_dir, save_name="model"
                    )
                else:
                    self.state_dict["model"].load_state_dict(
                        torch.load(self.resume_from_dir / "model.pt")
                    )
                if verbose:
                    print(f"Loading model state from model.pt")

            # load all of optimizer, scheduler, regularizer if they exist
            for module in ["optimizer", "scheduler", "regularizer"]:
                if module in saved_modules:
                    self.state_dict[module].load_state_dict(
                        torch.load(self.resume_from_dir / f"{module}.pt")
                    )

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
