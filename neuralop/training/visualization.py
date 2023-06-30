import io
import open3d as o3d
import numpy as np
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from PIL import Image


# Create a function to convert a figure to a NumPy array
def fig_to_numpy(fig: mpl.figure.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    im = Image.open(buf)
    im = np.array(im)
    buf.close()

    # Convert to valid image
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=-1)
    # if the image has 4 channels, remove the alpha channel
    if im.shape[-1] == 4:
        im = im[..., :3]
    # Convert to uint8 image
    if im.dtype != np.uint8:
        im = (im * 255).astype(np.uint8)
    return im


class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)[:, 0:3]


def vis_pressure(mesh_path, pressures, colormap="plasma", eps=0.5):
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    color_mapper = MplColorHelper(colormap, pressures.min(), pressures.max())
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        color_mapper.get_rgb(pressures[0, :])
    )

    meshes = [mesh]
    if pressures.shape[0] > 1:
        min_b = np.asarray(mesh.get_min_bound())
        max_b = np.asarray(mesh.get_max_bound())
        translation = np.array([max_b[0] - min_b[0] + eps, 0, 0])
        for j in range(1, pressures.shape[0]):
            new_mesh = o3d.io.read_triangle_mesh(mesh_path)
            new_mesh.translate(j * translation)
            new_mesh.vertex_colors = o3d.utility.Vector3dVector(
                color_mapper.get_rgb(pressures[j, :])
            )

            meshes.append(new_mesh)

    o3d.visualization.draw_geometries(meshes)
