import torch
import wandb
import sys
from neuralop.training import setup, AdamW
from neuralop import get_model
from neuralop.utils import get_wandb_api_key
from neuralop.losses.data_losses import LpLoss
from neuralop.training.trainer import Trainer
from neuralop.data.datasets import CarCFDDataset
from neuralop.data.transforms.data_processors import DataProcessor
from copy import deepcopy

# query points is [sdf_query_resolution] * 3 (taken from config ahmed)
# Read the configuration
config_name = 'cfd'
from zencfg import cfg_from_commandline
import sys 
sys.path.insert(0, '../')
from config.gino_carcfd_config import Default

config = cfg_from_commandline(Default)
config = config.to_dict()


#Set-up distributed communication, if using
device, is_logger = setup(config)

if config.data.sdf_query_resolution < config.model.fno_n_modes[0]:
    config.model.fno_n_modes = [config.data.sdf_query_resolution]*3

#Set up WandB logging
wandb_init_args = {}
config_name = 'car-pressure'
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = '_'.join(
            f'{var}' for var in [config_name, config.data.sdf_query_resolution])

    wandb_init_args = dict(config=config, 
                           name=wandb_name, 
                           group=config.wandb.group,
                           project=config.wandb.project,
                           entity=config.wandb.entity)

    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_init_args)

#Load CFD body data
data_module = CarCFDDataset(root_dir=config.data.root, 
                             query_res=[config.data.sdf_query_resolution]*3, 
                             n_train=config.data.n_train, 
                             n_test=config.data.n_test, 
                             download=config.data.download
                             )


train_loader = data_module.train_loader(batch_size=1, shuffle=True)
test_loader = data_module.test_loader(batch_size=1, shuffle=False)

model = get_model(config)

#Create the optimizer
optimizer = AdamW(model.parameters(), 
                                lr=config.opt.learning_rate, 
                                weight_decay=config.opt.weight_decay)

if config.opt.scheduler == 'ReduceLROnPlateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.opt.gamma, patience=config.opt.scheduler_patience, mode='min')
elif config.opt.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.opt.scheduler_T_max)
elif config.opt.scheduler == 'StepLR':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=config.opt.step_size,
                                                gamma=config.opt.gamma)
else:
    raise ValueError(f'Got {config.opt.scheduler=}')


l2loss = LpLoss(d=2,p=2)

if config.opt.training_loss == 'l2':
    train_loss_fn = l2loss
else: 
    raise ValueError(f'Got {config.opt.training_loss=}')

if config.opt.testing_loss == 'l2':
    test_loss_fn = l2loss
else:
    raise ValueError(f'Got {config.opt.testing_loss=}')

# Handle data preprocessing to gino 

class GINOCFDDataProcessor(DataProcessor):
    """
    Implements logic to preprocess data/handle model outputs
    to train an GINO on the CFD car-pressure dataset
    """

    def __init__(self, normalizer, device='cuda'):
        super().__init__()
        self.normalizer = normalizer
        self.device = device
        self.model = None

    def preprocess(self, sample):
        # Turn a data dictionary returned by MeshDataModule's DictDataset
        # into the form expected by the GINO
        
        # input geometry: just vertices
        in_p = sample['vertices'].squeeze(0).to(self.device)
        latent_queries = sample['query_points'].squeeze(0).to(self.device)
        out_p = sample['vertices'].squeeze(0).to(self.device)
        f = sample['distance'].to(self.device)

        #Output data
        truth = sample['press'].squeeze(0).unsqueeze(-1)

        # Take the first 3586 vertices of the output mesh to correspond to pressure
        # if there are less than 3586 vertices, take the maximum number of truth points
        output_vertices = truth.shape[1]
        if out_p.shape[0] > output_vertices:
            out_p = out_p[:output_vertices,:]

        truth = truth.to(device)

        batch_dict = dict(input_geom=in_p,
                          latent_queries=latent_queries,
                          output_queries=out_p,
                          latent_features=f,
                          y=truth,
                          x=None)

        sample.update(batch_dict)

        return sample
    
    def postprocess(self, out, sample):
        if not self.training:
            out = self.normalizer.inverse_transform(out)
            y = self.normalizer.inverse_transform(sample['y'].squeeze(0))
            sample['y'] = y

        return out, sample
    
    def to(self, device):
        self.device = device
        self.normalizer = self.normalizer.to(device)
        return self
    
    def wrap(self, model):
        self.model = model

    def forward(self, sample):
        sample = self.preprocess(sample)
        out = self.model(sample)
        out, sample = self.postprocess(out, sample)
        return out, sample

output_encoder = deepcopy(data_module.normalizers['press']).to(device)
data_processor = GINOCFDDataProcessor(normalizer=output_encoder, device=device)

trainer = Trainer(model=model, 
                  n_epochs=config.opt.n_epochs,
                  data_processor=data_processor,
                  device=device,
                  wandb_log=config.wandb.log,
                  verbose=is_logger
                  )

if config.wandb.log:
    wandb.log({'time_to_distance': data_module.time_to_distance}, commit=False)

trainer.train(
              train_loader=train_loader,
              test_loaders={'test':test_loader},
              optimizer=optimizer,
              scheduler=scheduler,
              training_loss=train_loss_fn,
              eval_losses={config.opt.testing_loss: test_loss_fn},
              regularizer=None,)