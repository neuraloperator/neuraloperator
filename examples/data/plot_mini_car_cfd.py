"""
Visualizing computational fluid dynamics on a car
===================================================
In this example we visualize a mesh from the `mini_car` dataset provided along with the
"""

# %%
# Import dependencies
# --------------------
# We first import our `neuralop` library and required dependencies.
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from neuralop.data.datasets import load_mini_car

font = {'size'   : 28}
matplotlib.rc('font', **font)

torch.manual_seed(0)
np.random.seed(0)

# %%
# Understanding the data 
# ----------------------
# The data in a ``MeshDataModule`` is structured as a dictionary of tensors and important scalar values encoding 
# a 3-d triangle mesh over the surface of a car. 
# Each sample includes the coordinates of all triangle vertices and the centroids of each triangle face.
# In this case, the creators used OpenFOAM to simulate the surface air pressure on car geometries in a wind tunnel. 
# The 3-d Navier-Stokes equations were simulated for a variety of inlet velocities over each surface using the 
# **OpenFOAM** computational solver to predict pressure at every vertex on the mesh. 
# Each sample here also has an inlet velocity scalar and a pressure field that maps 1-to-1 with the vertices on the mesh.
dataset = load_mini_car(n_train=3, n_test=1, query_res=[16,16,16])
sample = dataset.train_data[0]
print(f'{sample.keys()=}')

# %%
# Visualizing the data 
# ----------------------
# We first define a utility function to plot our point clouds:

def draw_car_view(coords, colors=None):
    '''
    Draw a properly rotated and scaled view of a car point cloud
    
    Parameters
    ----------
    coords: torch.Tensor
        coordinates of the points to plot, shape (1, n_pts, 3)
    colors: torch.Tensor, optional
        optional field mapping 1-to-1 with the coords above, shape (1,n_pts,)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax.view_init(azim=-60, dist=200)
    scatter = ax.scatter(coords[0,:,0],coords[0,:,1],coords[0,:,2], s=2, c=colors)
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=20, azim=150, roll=0, vertical_axis='y')
    ax.set_title("Pressure over car mesh vertices")
    fig.colorbar(scatter, fraction=0.02, pad=0.2, label="normalized pressure", ax=ax)
    plt.draw()

draw_car_view(sample['vertices'], sample['press'])
# %%
# 