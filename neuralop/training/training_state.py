"""
Snippet to load all artifacts of training state as Modules
without constraining to use inside a default Trainer
"""
from typing import Union
from pathlib import Path

import torch
from torch import nn


def load_training_state(save_dir: Union[str, Path], save_name: str,
                        model: nn.Module,
                        optimizer: nn.Module = None,
                        scheduler: nn.Module = None,
                        regularizer: nn.Module = None) -> dict:
    """load_training_state returns model and optional other training modules
    saved from prior training for downstream use

    Parameters
    ----------
    save_dir : Union[str, Path]
        directory from which to load training state (model, optional optimizer, scheduler, regularizer)
    save_name : str
        name of model to load
    """
    training_state = {}

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    
    training_state['model'] = model.from_checkpoint(save_dir, save_name)
    
    # load optimizer if state exists
    if optimizer is not None:
        optimizer_pth = save_dir / "optimizer.pt"
        if optimizer_pth.exists():
            training_state['optimizer'] = optimizer.load_state_dict(torch.load(optimizer_pth))
        else:
            print(f"Warning: requested to load optimizer state, but no saved optimizer state exists in {save_dir}.")
    
    if scheduler is not None:
        scheduler_pth = save_dir / "scheduler.pt"
        if scheduler_pth.exists():
            training_state['scheduler'] = scheduler.load_state_dict(torch.load(scheduler_pth))
        else:
            print(f"Warning: requested to load scheduler state, but no saved scheduler state exists in {save_dir}.")
    
    if regularizer is not None:
        regularizer_pth = save_dir / "regularizer.pt"
        if regularizer_pth.exists():
            training_state['regularizer'] = scheduler.load_state_dict(torch.load(regularizer_pth))
        else:
            print(f"Warning: requested to load regularizer state, but no saved regularizer state exists in {save_dir}.")
    
    return training_state


def save_training_state(save_dir: Union[str, Path], save_name: str,
                        model: nn.Module,
                        optimizer: nn.Module = None,
                        scheduler: nn.Module = None,
                        regularizer: nn.Module = None) -> None:
    """save_training_state returns model and optional other training modules
    saved from prior training for downstream use

    Parameters
    ----------
    save_dir : Union[str, Path]
        directory from which to load training state (model, optional optimizer, scheduler, regularizer)
    save_name : str
        name of model to load
    """
    training_state = {}

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    
    training_state['model'] = model.save_checkpoint(save_dir, save_name)
    
    # load optimizer if state exists
    if optimizer is not None:
        optimizer_pth = save_dir / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_pth)
    
    if scheduler is not None:
        scheduler_pth = save_dir / "scheduler.pt"
        torch.save(scheduler.state_dict(), scheduler_pth)
    
    if regularizer is not None:
        regularizer_pth = save_dir / "regularizer.pt"
        torch.save(regularizer.state_dict(), regularizer_pth)
    
    print(f"Successfully saved training state to {save_dir}")