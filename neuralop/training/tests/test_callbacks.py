import os
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from neuralop.training import Trainer, LpLoss, H1Loss, CheckpointCallback

class DummyDataset(Dataset):
    # Simple linear regression problem, PyTorch style

    def __init__(self, n_examples):
        super().__init__()

        self.X = torch.randn((n_examples, 50))
        self.y = torch.randn((n_examples, 1))

    def __getitem__(self, idx):
        return {'x': self.X[idx], 'y': self.y[idx]}

    def __len__(self):
        return self.X.shape[0]

class DummyModel(nn.Module):
    """
    Simple linear model to mock-up our model API
    """

    def __init__(self, features):
        super().__init__()
        self.net = nn.Linear(features, 1)

    def forward(self, x, **kwargs):
        """
        Throw out extra args as in FNO and other models
        """
        return self.net(x)

def test_model_checkpoint_saves():
    model = DummyModel(50)

    train_loader = DataLoader(DummyDataset(100))

    trainer = Trainer(model=model,
                      n_epochs=5,
                      callbacks=[
                          CheckpointCallback(save_dir='./checkpoints',
                                             save_optimizer=True,
                                             save_scheduler=True)
                      ]
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
                  )
    
    assert sorted(os.listdir('./checkpoints')) == sorted(['model.pt', 'optimizer.pt', 'scheduler.pt'])


def test_model_checkpoint_and_resume():
    model = DummyModel(50)

    train_loader = DataLoader(DummyDataset(100))
    test_loader = DataLoader(DummyDataset(20))

    trainer = Trainer(model=model,
                      n_epochs=5,
                      callbacks=[
                          CheckpointCallback(save_dir='./full_states',
                                                  save_optimizer=True,
                                                  save_scheduler=True,
                                                  save_best='h1') # monitor h1 loss
                      ]
    )

    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=8e-3, 
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Creating the losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    eval_losses={'h1': h1loss, 'l2': l2loss}

    trainer.train(train_loader=train_loader, 
                  test_loaders={'': test_loader}, 
                  optimizer=optimizer,
                  scheduler=scheduler,
                  regularizer=None,
                  training_loss=l2loss,
                  eval_losses=eval_losses
                  )
    
    assert sorted(os.listdir('./full_states')) == sorted(['best_model.pt', 'optimizer.pt', 'scheduler.pt'])

    # Resume from checkpoint
    trainer = Trainer(model=model,
                      n_epochs=5,
                      callbacks=[
                          CheckpointCallback(save_dir='./checkpoints',
                                             resume_from_dir='./full_states')
                          ]
    )

    trainer.train(train_loader=train_loader, 
                  test_loaders={'': test_loader}, 
                  optimizer=optimizer,
                  scheduler=scheduler,
                  regularizer=None,
                  training_loss=l2loss,
                  eval_losses=eval_losses,
                  )
