import torch
import wandb
import sys
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from neuralop.training import setup
from neuralop import get_model
from neuralop.utils import get_wandb_api_key, count_params
from neuralop.training.losses import IregularLpqLoss, pressure_drag, friction_drag, WeightedL2DragLoss, FieldwiseAggregatorLoss, SumAggregatorLoss
from neuralop.training.trainer import Trainer
from neuralop.datasets.cfd_dataset import load_
from neuralop.datasets.output_encoder import MultipleFieldOutputEncoder
from copy import deepcopy
from timeit import default_timer


    
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
decoder_return_indices = {k:tuple([slice(*tuple(x)) for x in v]) for k,v in config.data.decoder_return_indices.items()}

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

#Load Ahmed body data
preprocessor = AhmedDataDictPreprocessor(config=config,
                                         device=device)
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


output_encoders = {}
for field in output_indices.keys():
    output_encoders[field] = deepcopy(data_module.normalizers[field]).to(device)

encoder_wrapper = MultipleFieldOutputEncoder(encoder_dict=output_encoders, input_mappings = output_indices, return_mappings = decoder_return_indices)

if config.opt.training_loss == 'l2':
    loss_fn = lambda x, y, z : torch.sqrt(torch.sum((x - y)**2) / torch.sum(y**2))
    train_loss_fn = FieldwiseAggregatorLoss({field: loss_fn for field in output_indices.keys()})
elif config.opt.training_loss == 'weightedl2' or config.opt.training_loss == 'weightedl2drag':
    drag_loss = WeightedL2DragLoss(device=device, mappings=output_indices)
    fieldwise_lpq = FieldwiseAggregatorLoss(losses={field: IregularLpqLoss() for field in output_indices.keys()}, mappings=output_indices)
    train_loss_fn = SumAggregatorLoss(drag_loss, fieldwise_lpq)
else:
    raise ValueError(f'Got {config.opt.training_loss=}')

# Handle Drag Error in validation separately for now
DragLoss = WeightedL2DragLoss(mappings = output_indices, device=device)

if config.opt.testing_loss == 'l2':
    field_loss = lambda x, y, z : torch.sqrt(torch.sum((x - y)**2) / torch.sum(y**2))
    test_loss_fn = FieldwiseAggregatorLoss(losses={field: field_loss for field in output_indices.keys()}, mappings=output_indices, logging=True)
elif config.opt.testing_loss == 'weightedl2':
    test_loss_fn = FieldwiseAggregatorLoss(losses={field: IregularLpqLoss() for field in output_indices.keys()}, mappings=output_indices, logging=True)
else:
    raise ValueError(f'Got {config.opt.testing_loss=}')

trainer = Trainer(model=model, 
                  n_epochs=config.opt.n_epochs,
                  output_field_indices=output_indices,
                  device=device,
                  sample_max=config.sample_max,
                  )

trainer.train(model=model,
              train_loader=train_loader,
              test_loaders=test_loader,
              output_encoder=encoder_wrapper,
              optimizer=optimizer,
              scheduler=scheduler,
              training_loss=train_loss_fn,
              eval_losses=test_loss_fn,
              regularizer=None,
              dataset_preprocess_fn=preprocessor)

model = model.cpu()