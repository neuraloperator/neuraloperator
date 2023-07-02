import torch

from neuralop.datasets.mesh_datamodule import MeshDataModule
from neuralop.training.losses import total_drag, IregularLpqLoss
from neuralop.models.integral_transform import IntegralTransform, NeighborSearch

#####TEST LOADING DATA#########

#Load ahmed body
data_mod = MeshDataModule('~/HDD/ahmed', 'case', 
                          query_points=[64,64,64], 
                          n_train=10, 
                          n_test=5, 
                          attributes=['pressure', 'wall_shear_stress', 'inlet_velocity', 'info', 'drag_history'])


#First training data example
example = data_mod.train_data[0]


#####TEST COMPUTING DRAG#################

# Quantities needed to compute drag
tri_areas = example['triangle_areas']
inward_normals = -example['triangle_normals']
flow_normals = torch.zeros((tri_areas.shape[0], 3))
flow_normals[:,0] = -1.0
pressure = example['pressure']
stress = example['wall_shear_stress']

flow_speed = example['info']['velocity']
reference_w = example['info']['width']
reference_h = example['info']['height']

#Undo normalization on pressure and stress
true_pressure = data_mod.normalizers['pressure'].decode(pressure)
true_stress = data_mod.normalizers['wall_shear_stress'].decode(stress)

#Compute drag 
computed_drag = total_drag(true_pressure,
                           true_stress,
                           tri_areas,
                           inward_normals,
                           flow_normals,
                           flow_speed,
                           (reference_w*reference_h/2) * 1e-6)

#Reference drag
reference_drag = example['drag_history']['c_p'][-1] + \
                 example['drag_history']['c_f'][-1]

print(f'Computed drag: {computed_drag}')
print(f'Reference drag: {reference_drag}')

#####GNO TEST#################

nb_search = NeighborSearch()

out_channels = 10
in_channels = 6 + out_channels #dim x + dim y + dim f (will change based on transform type)
#t0: dim x + dim y = 6
#t1/2 : dim x + dim y + dim f
#t0/2 : out_channels = dim f
radius = 0.016 #smaller than before since query points on [0,1]^3 not [-1,1]^3
gno = IntegralTransform(mlp_layers=[in_channels,512,256,out_channels],
                        transform_type=2)

# x = x_in : (n_in, 3)
# y = x_out : (n_x, n_y, n_z, 3)
# fy = df : (1, n_x, n_y, n_z)

#SDF query points
y = example['query_points']
#Surface triangle centroids
x = example['centroids']
#Some function on the query point (output of FNO)
f_y = torch.randn(64,64,64,out_channels)

#Compute neighbors
neighbors = nb_search(y.view(-1,3), radius, x)

#Integrate 
gno_out = gno(y=y.view(-1,3),
              neighbors=neighbors, 
              x=x,
              f_y=f_y.view(-1,out_channels))

print(f'GNO out: {gno_out.shape}')

#####TEST WEIGHTED LOSS#################

loss = IregularLpqLoss()

#Take arbirary components as prediction
pred_pressure = gno_out[:,0]
pred_stress = gno_out[:,1:4]

pressure_error = loss(pred_pressure, pressure, tri_areas)
stress_error = loss(pred_stress, stress, tri_areas)

print(f'Relative pressure error: {pressure_error}')
print(f'Relative stress error: {stress_error}')


# python ahmed_test.py