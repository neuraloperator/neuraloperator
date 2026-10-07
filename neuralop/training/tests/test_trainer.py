import os
import shutil
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from neuralop.models import FNO
from neuralop.data.datasets import load_darcy_flow_small

from neuralop import Trainer, LpLoss, H1Loss
from neuralop.tests.test_utils import DummyDataset, DummyModel
from neuralop.training import IncrementalFNOTrainer, AdamW


def test_model_checkpoint_saves():
    save_pth = Path("./test_checkpoints")

    model = DummyModel(50)

    train_loader = DataLoader(DummyDataset(100))

    trainer = Trainer(
        model=model,
        n_epochs=5,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Creating the losses
    l2loss = LpLoss(d=2, p=2)

    trainer.train(
        train_loader=train_loader,
        test_loaders={},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=l2loss,
        eval_losses=None,
        save_dir=save_pth,
        save_every=1,
    )

    for file_ext in [
        "model_state_dict.pt",
        "model_metadata.pkl",
        "optimizer.pt",
        "scheduler.pt",
    ]:
        file_pth = save_pth / file_ext
        assert file_pth.exists()

    # clean up dummy checkpoint directory after testing
    shutil.rmtree("./test_checkpoints")


def test_model_checkpoint_and_resume():
    save_pth = Path("./full_states")
    model = DummyModel(50)

    train_loader = DataLoader(DummyDataset(100))
    test_loader = DataLoader(DummyDataset(20))

    trainer = Trainer(model=model, n_epochs=5, verbose=True)

    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Creating the losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    eval_losses = {"h1": h1loss, "l2": l2loss}

    trainer.train(
        train_loader=train_loader,
        test_loaders={"test": test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=l2loss,
        eval_losses=eval_losses,
        save_best="test_h1",
        save_dir=save_pth,
        save_every=1,
    )
    
    for file_ext in [
        "best_model_state_dict.pt",
        "best_model_metadata.pkl",
        "optimizer.pt",
        "scheduler.pt",
    ]:
        file_pth = save_pth / file_ext

        assert file_pth.exists()

    # Resume from checkpoint

    new_model = DummyModel(50)
    new_optimizer = AdamW(new_model.parameters(), lr=3e-4, weight_decay=1e-4)
    trainer = Trainer(model=new_model, n_epochs=10, verbose=True)
    errors = trainer.train(
        train_loader=train_loader,
        test_loaders={"": test_loader},
        optimizer=new_optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=l2loss,
        eval_losses=eval_losses,
        resume_from_dir=save_pth,
    )

    # Ensure model and opt parameter IDs match after reloading - otherwise model will not train

    # Get model and optimizer parameters as a set of IDs
    model_param_ids = {id(p) for p in trainer.model.parameters()}
    optimizer_param_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}

    # Check for mismatches, assert there are none
    missing_in_optimizer = model_param_ids - optimizer_param_ids
    missing_in_model = optimizer_param_ids - model_param_ids

    assert not missing_in_optimizer and not missing_in_model

    # clean up dummy checkpoint directory after testing
    shutil.rmtree(save_pth)


# ensure that model accuracy after loading from checkpoint
# is comparable to accuracy at time of save
def test_load_from_checkpoint():
    model = DummyModel(50)

    train_loader = DataLoader(DummyDataset(100))
    test_loader = DataLoader(DummyDataset(100))

    trainer = Trainer(model=model, n_epochs=10)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Creating the losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    eval_losses = {"h1": h1loss, "l2": l2loss}

    orig_model_eval_errors = trainer.train(
        train_loader=train_loader,
        test_loaders={"test": test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=l2loss,
        eval_losses=eval_losses,
        save_dir="./full_states",
        save_every=1,
    )

    # create a new model from saved checkpoint and evaluate
    loaded_model = DummyModel.from_checkpoint(save_folder="./full_states", save_name="model")
    trainer = Trainer(model=loaded_model, n_epochs=1)

    loaded_model_eval_errors = trainer.evaluate(
        loss_dict=eval_losses, data_loader=test_loader, log_prefix="test"
    )

    # test l2 difference should be small
    assert (
        orig_model_eval_errors["test_l2"] - loaded_model_eval_errors["test_l2"]
    ) / orig_model_eval_errors["test_l2"] < 0.1

    # clean up dummy checkpoint directory after testing
    shutil.rmtree("./full_states")


# enure that the model incrementally increases in frequency modes
def test_incremental():
    # Loading the Darcy flow dataset
    train_loader, test_loaders, output_encoder = load_darcy_flow_small(
        n_train=10,
        batch_size=16,
        test_resolutions=[16, 32],
        n_tests=[10, 5],
        test_batch_sizes=[32, 32],
    )

    initial_n_modes = (2, 2)
    initial_max_modes = (16, 16)

    model = FNO(
        n_modes=initial_n_modes,
        max_n_modes=initial_max_modes,
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
    )

    trainer = IncrementalFNOTrainer(
        model=model,
        n_epochs=20,
        incremental_loss_gap=False,
        incremental_grad=True,
        incremental_grad_eps=0.9999,
        incremental_buffer=5,
        incremental_max_iter=1,
        incremental_grad_max_iter=2,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Creating the losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    eval_losses = {"h1": h1loss, "l2": l2loss}

    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=l2loss,
        eval_losses=eval_losses,
    )

    # assert that the model has increased in frequency modes
    for i in range(len(initial_n_modes)):
        assert model.fno_blocks.convs[0].n_modes[i] > initial_n_modes[i]

    # assert that the model has not changed the max modes
    for i in range(len(initial_max_modes)):
        assert model.fno_blocks.convs[0].max_n_modes[i] == initial_max_modes[i]

# ---------------------------------------------------------------------------
# Tests for autoregressive evaluation in Trainer.evaluate.
# Checks that both mode spellings ("autoregression" / "autoregressive") work,
# and that invalid inputs (unknown mode, max_steps=None without a
# data_processor, max_steps <= 0) raise a clear ValueError instead of
# crashing, looping forever, or dividing by zero.
# ---------------------------------------------------------------------------

# A tiny model: one linear layer, 4 inputs -> 4 outputs.
class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x, **kwargs):
        return self.lin(x)

# A tiny dataset: 4 samples of random numbers in the {"x": ..., "y": ...}
class _TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        return {"x": torch.randn(4), "y": torch.randn(4)}

# Helper that builds a fresh Trainer and DataLoader for each test, so the tests don't repeat the same setup code.
def _make_trainer_and_loader():
    trainer = Trainer(model=_TinyModel(), n_epochs=1, device="cpu", verbose=False)
    loader = DataLoader(_TinyDataset(), batch_size=2)
    return trainer, loader

# Both the correct mode name and the spelling used in the docs
# ("autoregressive") should run without errors.
@pytest.mark.parametrize("mode", ["autoregression", "autoregressive"])
def test_autoregressive_eval_runs(mode):
    trainer, loader = _make_trainer_and_loader()
    errors = trainer.evaluate(
        {"l2": LpLoss(d=1, p=2)}, loader, log_prefix="test", mode=mode, max_steps=3
    )
    assert "test_l2" in errors

# An unknown mode should raise a clear ValueError, not a confusing UnboundLocalError.
def test_eval_unknown_mode_raises():
    trainer, loader = _make_trainer_and_loader()
    with pytest.raises(ValueError, match="Unknown evaluation mode"):
        trainer.evaluate({"l2": LpLoss(d=1, p=2)}, loader, mode="rollout")

# Without a data_processor and without max_steps, nothing can stop the rollout loop. 
# This should raise a ValueError instead of looping forever.
def test_autoregressive_eval_without_stop_condition_raises():
    trainer, loader = _make_trainer_and_loader()
    with pytest.raises(ValueError, match="requires max_steps"):
        trainer.evaluate(
            {"l2": LpLoss(d=1, p=2)}, loader, mode="autoregression", max_steps=None
        )

# max_steps of 0 or a negative number should raise a ValueError instead of dividing by zero.
@pytest.mark.parametrize("max_steps", [0, -1])
def test_autoregressive_eval_nonpositive_steps_raises(max_steps):
    trainer, loader = _make_trainer_and_loader()
    with pytest.raises(ValueError, match="max_steps must be a positive integer"):
        trainer.evaluate(
            {"l2": LpLoss(d=1, p=2)},
            loader,
            mode="autoregression",
            max_steps=max_steps,
        )