================================
Training neural operator models
================================

Our library makes it easy for anyone with data drawn from a system governed by a PDE to train and test Neural Operator models. 
This page details the library's Python interface for training and evaluation of NOs.


The Trainer class
==================
Most users will train neural operator models on their own data in very similar ways, 
using a very standard machine learning training loop. To speed up this process, we 
provide a :code:`Trainer` class that automates much of this boilerplate logic. 
Using the small Darcy flow dataset we provide along with the library, you can use a 
:code:`Trainer` as follows:

.. code:: python
    import torch

    from neuralop.models import FNO2d
    from neuralop.datasets import load_darcy_flow_small
    from neuralop.training import Trainer, LpLoss, H1Loss

    # Load the dataset
    train_loader, test_loaders, output_encoder = load_darcy_flow_small(
        n_train=100, batch_size=4, 
        test_resolutions=[16, 32], n_tests=[50, 50], test_batch_sizes=[4, 2],
        )
    
    # Load the model
    model = FNO2d(n_modes=16,
                  hidden_channels=32)
    
    # Create losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    train_loss = l2loss
    eval_losses = {'h1':h1loss, 'l2':l2loss}

    # Configure optimizer
    optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
    )

    # Configure scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=5,
        mode="min",
    )

    # Create Trainer object
    trainer = Trainer(
        model=model,
        n_epochs=100
    )

    # This is all you need to do to fit the model!
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=train_loss,
        eval_losses=eval_losses
    )


If you need to implement more domain-specific training logic, we also expose an
interface for stateful Callbacks that trigger events and track information
throughout the lifetime of your Trainer. For instance, if you wanted to save 
model weights during  training in the above script, you would simply modify these lines:

.. code:: python
    from neuralop.training import Trainer, ModelCheckpointCallback, LpLoss, H1Loss

    ## ... code continues as above

    trainer = Trainer(
        model=model,
        n_epochs=100,
        callbacks=[
            ModelCheckpointCallback(
                checkpoint_dir='./checkpoints',
                interval=5 # save a checkpoint every 5 epochs
                )
            ]
    )

    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=train_loss,
        eval_losses=eval_losses
    )

...and that's it! For more specific documentation on callbacks, check the API reference.
