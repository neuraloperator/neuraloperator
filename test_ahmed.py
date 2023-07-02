import torch

from neuralop.datasets.mesh_datamodule import MeshDataModule
from neuralop.training.losses import total_drag, IregularLpqLoss
from neuralop.models.integral_transform import IntegralTransform, NeighborSearch
from neuralop.models.FNOGNO import FNOGNO
from neuralop.training.losses import total_drag, IregularLpqLoss, LpLoss


data_mod = MeshDataModule('~/projects/neuraloperator_run/data/new_ahmed', 'case', 
                          query_points=[64,64,64], 
                          n_train=10, 
                          n_test=5, 
                          attributes=['pressure', 'wall_shear_stress', 'inlet_velocity', 'info', 'drag_history'])

example = data_mod.train_data[0]

model = FNOGNO()
loss_fn = LpLoss()
model_out = model.eval_dict(example, loss_fn=loss_fn)
print('FNOGNOAhmed loss2:{}'.format(model_out['l2']))
