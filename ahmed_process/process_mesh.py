import pyvista as pv
import os 
import numpy as np
import torch

path = 'data/'

dirs = [d for d in os.listdir(path) if os.path.isdir(path + d)]

for dir in dirs:
    mesh = pv.read(path + dir + '/ahmed_body.vtp')
    mesh = mesh.extract_surface().triangulate()

    mesh.save(path + dir + '/tri_mesh.ply')

    pressure = torch.tensor(np.asarray(mesh['p']), dtype=torch.float32)
    stress = torch.tensor(np.asarray(mesh['wallShearStress']), dtype=torch.float32)

    torch.save(pressure, path + dir + '/pressure.pt')
    torch.save(stress, path + dir + '/wall_shear_stress.pt')

    print(dir)