import os
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from neuralop.models import FNO
from neuralop.data.datasets import load_darcy_flow_small

from neuralop import Trainer, LpLoss, H1Loss
from neuralop.tests.test_utils import DummyDataset, DummyModel

def test_model_checkpoint_saves():
    save_pth = Path('./test_checkpoints')

    model = DummyModel(50)

    train_loader = DataLoader(DummyDataset(100))

    trainer = Trainer(model=model,
                      n_epochs=5,
    )

    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=8e-3, 
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Creating the losses
    l2loss = LpLoss(d=2, p=2)

    trainer.train(train_loader=train_loader, 
                  test_loaders={}, 
                  optimizer=optimizer,
                  scheduler=scheduler,
                  regularizer=None,
                  training_loss=l2loss,
                  eval_losses=None,
                  save_dir=save_pth,
                  save_every=1
                  )
    
    for file_ext in ['model_state_dict.pt', 'model_metadata.pkl', 'optimizer.pt', 'scheduler.pt']:
        file_pth = save_pth / file_ext
        assert file_pth.exists()

    # clean up dummy checkpoint directory after testing
    shutil.rmtree('./test_checkpoints')

def test_model_checkpoint_and_resume():
    save_pth = Path('./full_states')
    model = DummyModel(50)

    train_loader = DataLoader(DummyDataset(100))
    test_loader = DataLoader(DummyDataset(20))

    trainer = Trainer(model=model,
                      n_epochs=5,
    )

    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=3e-4, 
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Creating the losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    eval_losses={'h1': h1loss, 'l2': l2loss}

    trainer.train(train_loader=train_loader, 
                  test_loaders={'test': test_loader}, 
                  optimizer=optimizer,
                  scheduler=scheduler,
                  regularizer=None,
                  training_loss=l2loss,
                  eval_losses=eval_losses,
                  save_best='test_h1',
                  save_dir=save_pth,
                  save_every=1
                  )
    for file_ext in ['best_model_state_dict.pt', 'best_model_metadata.pkl', 'optimizer.pt', 'scheduler.pt']:
        file_pth = save_pth / file_ext
        
        assert file_pth.exists()

    # Resume from checkpoint
    trainer = Trainer(model=model,
                      n_epochs=5
    )
    errors = trainer.train(train_loader=train_loader, 
                  test_loaders={'': test_loader}, 
                  optimizer=optimizer,
                  scheduler=scheduler,
                  regularizer=None,
                  training_loss=l2loss,
                  eval_losses=eval_losses,
                  resume_from_dir=save_pth
                  )
    
    # clean up dummy checkpoint directory after testing
    shutil.rmtree(save_pth)

    
# ensure that model accuracy after loading from checkpoint
# is comparable to accuracy at time of save
def test_load_from_checkpoint():
    model = DummyModel(50)

    train_loader = DataLoader(DummyDataset(100))
    test_loader = DataLoader(DummyDataset(100))

    trainer = Trainer(model=model,
                      n_epochs=10,)

    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=3e-4, 
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Creating the losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    eval_losses={'h1': h1loss, 'l2': l2loss}

    orig_model_eval_errors = trainer.train(train_loader=train_loader, 
                  test_loaders={'test': test_loader}, 
                  optimizer=optimizer,
                  scheduler=scheduler,
                  regularizer=None,
                  training_loss=l2loss,
                  eval_losses=eval_losses,
                  save_dir="./full_states",
                  save_best="test_l2"
                  )
    
    # create a new model from saved checkpoint and evaluate
    loaded_model = DummyModel.from_checkpoint(save_folder='./full_states', save_name='best_model')
    trainer = Trainer(model=loaded_model,
                      n_epochs=1,
    )

    loaded_model_eval_errors = trainer.evaluate(loss_dict=eval_losses,
                              data_loader=test_loader, log_prefix='test')

    # log prefix is empty except for default underscore
    assert orig_model_eval_errors['test_l2'] - loaded_model_eval_errors['test_l2'] < 0.1

    # clean up dummy checkpoint directory after testing
    shutil.rmtree('./full_states')
    