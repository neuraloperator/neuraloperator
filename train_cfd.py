import torch
import wandb
import sys
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from neuralop.training import setup
from neuralop import get_model
from neuralop.utils import get_wandb_api_key
from neuralop.training.losses import IregularLpqLoss, WeightedL2DragLoss, SumAggregatorLoss
from neuralop.training.trainer import Trainer
from neuralop.datasets.mesh_datamodule import MeshDataModule
from copy import deepcopy
from neuralop.training.callbacks import NoPatchingOutputEncoderCallback, Callback

# query points is [sdf_query_resolution] * 3 (taken from config ahmed)

    
# Read the configuration
config_name = 'cfd'
pipe = ConfigPipeline([YamlConfig('./cfd_config.yaml', config_name=config_name, config_folder='config'),
                       ArgparseConfig(infer_types=True, config_name=None, config_file=None),
                       YamlConfig(config_folder='config')
                      ])
config = pipe.read_conf()

#Set-up distributed communication, if using
device, is_logger = setup(config)

if config.data.sdf_query_resolution < config.fnogno.fno_n_modes[0]:
    config.fnogno.fno_n_modes = [config.data.sdf_query_resolution]*3

# output indices follow form in train_ahmed
output_indices = {k:tuple([slice(*tuple(x)) for x in v]) for k,v in config.data.output_indices.items()}

# only some channels of output encoder's decoded data
# are grabbed in train_ahmed original

#Set up WandB logging
config_name = 'dragloss'
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = '_'.join(
            f'{var}' for var in [config_name, config.data.sdf_query_resolution])
        
    wandb.init(config=config, name=wandb_name, group=config.wandb.group,
               project=config.wandb.project, entity=config.wandb.entity)
    
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]

#Load CFD body data
data_module = MeshDataModule(config.data.path, 
                             config.data.entity_name, 
                             query_points=[config.data.sdf_query_resolution]*3, 
                             n_train=config.data.n_train, 
                             n_test=config.data.n_test, 
                             attributes=config.data.load_attributes,
                             )

if config.wandb.log:
    wandb.log({'time_to_distance': data_module.time_to_distance})

train_loader = data_module.train_dataloader(batch_size=1, shuffle=True)
test_loader = data_module.test_dataloader(batch_size=1, shuffle=False)

model = get_model(config)
model = model.to(device)

#Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
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

output_encoder = deepcopy(data_module.normalizers['press']).to(device)

if config.opt.training_loss == 'l2':
    train_loss_fn = lambda x, y, z : torch.sqrt(torch.sum((x - y)**2) / torch.sum(y**2))
elif config.opt.training_loss == 'weightedl2' or config.opt.training_loss == 'weightedl2drag':
    drag_loss = WeightedL2DragLoss(device=device, mappings=output_indices)
    train_loss_fn = SumAggregatorLoss(drag_loss, IregularLpqLoss())
else:
    raise ValueError(f'Got {config.opt.training_loss=}')

# Handle Drag Error in validation separately for now
DragLoss = WeightedL2DragLoss(mappings = output_indices, device=device)

if config.opt.testing_loss == 'l2':
    test_loss_fn = lambda x, y, z : torch.sqrt(torch.sum((x - y)**2) / torch.sum(y**2))
elif config.opt.testing_loss == 'weightedl2':
    test_loss_fn = IregularLpqLoss()
else:
    raise ValueError(f'Got {config.opt.testing_loss=}')

encoder_callback = NoPatchingOutputEncoderCallback(encoder=output_encoder)

# Handle data preprocessing to FNOGNO as a Callback object
        
class CFDDataPreprocessingCallback(Callback):
    """
    Implements logic to preprocess data/handle model outputs
    to train an FNOGNO on the CFD car-pressure dataset
    """

    def __init__(self):
        super().__init__()

    def on_batch_start(self, idx, sample):
        self.state_dict['sample'] = sample
        device = self.state_dict.get('device', 'cpu')
        in_p = sample['query_points'].squeeze(0).to(device)
        out_p = sample['centroids'].squeeze(0).to(device)
        f = sample['distance'].squeeze(0).to(device)
        
        weights = sample['triangle_areas'].squeeze(0).to(device)

        #Output data
        truth = sample['press'].squeeze(0)
        if len(truth.shape) == 1:
            truth = truth.unsqueeze(-1)
        
        truth = truth.to(device)

        inward_normals = -sample['triangle_normals'].squeeze(0).to(device)
        flow_normals = torch.zeros((sample['triangle_areas'].shape[1], 3)).to(device)
        flow_normals[:,0] = -1.0

        '''
        The current cleaned CFD dataset doesn't contain these.
        
        Revisit when preprocessing again, as they are part of another
        MeshDataModule dataset.

        flow_speed = sample['info']['velocity'].item()
        reference_w = sample['info']['width'].item()
        reference_h = sample['info']['height'].item()
        reference_area = (reference_w*reference_h/2) * 1e-6
        '''


        self.state_dict['sample'].update(in_p = in_p,
                                         out_p=out_p,
                                         f=f,
                                         y=truth,
                                         inward_normals=inward_normals,
                                         flow_normals=flow_normals,
                                         flow_speed=None,
                                         vol_elm=weights,
                                         reference_area=None)

trainer = Trainer(model=model, 
                  n_epochs=config.opt.n_epochs,
                  device=device,
                  callbacks=[
                      encoder_callback,
                      CFDDataPreprocessingCallback()
                  ]
                  )

trainer.train(
              train_loader=train_loader,
              test_loaders=test_loader,
              optimizer=optimizer,
              scheduler=scheduler,
              training_loss=train_loss_fn,
              #eval_losses={config.opt.testing_loss: test_loss_fn, 'drag': DragLoss},
              eval_losses=config.opt.testing_loss,
              regularizer=None,)

model = model.cpu()