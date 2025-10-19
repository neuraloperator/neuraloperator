from pathlib import Path
from timeit import default_timer
from typing import List, Union

import numpy as np

# import open3d for io if built. Otherwise,
# the class will build, but no files will be loaded.
try:
    import open3d as o3d
    o3d_warn = False
except ModuleNotFoundError:
    o3d_warn = True

try:
    import ot
    from ot.bregman import empirical_sinkhorn2_geomloss
    ot_warn = False
except ModuleNotFoundError:
    ot_warn = True   

import torch

# Get the directory of the current script
script_dir = Path(__file__).parent

class OTDataModule:
    def __init__(
        self,
        root_dir: Union[str, Path],
        item_dir_name: Union[str, Path],
        n_total: int = None,
        expand_factor: float = 3.0, 
        reg: float = 1e-06,
        device: Union[str, torch.device] = 'cuda',
    ):
        """OTDataModule generate OT data from physical mesh

        Parameters
        ----------
        root_dir : Union[str, Path]
            str or Path to root directory of CFD dataset
        item_dir_name : Union[str, Path]
            directory in which individual item subdirs are stored
        n_total : int, optional
            hard limit on number of total examples
            if n_total is greater than the actual number
            of total examples available, nothing is changed
        expand_factor : float, optional
            Scale factor to map physical mesh size to latent mesh size (e.g., torus/sphere).
            Affects OT plan surjectivity: smaller values may lead to incomplete mappings, while larger values increase computational cost but improve surjectivity. 
            Choose a value balancing accuracy and efficiency, by default 3.
        reg : float, optional
            Regularization coefficient for the Sinkhorn algorithm.
            Affects OT plan surjectivity: smaller values increase precision (where fewer non-surjective plans indicate higher precision) but incur higher computational cost.
            Choose a value balancing accuracy and efficiency, by default 1e-06.
        device : Union[str, torch.device], optional
            Device for OT computation.
        """
        self.device = device

        if o3d_warn or ot_warn:
            print("Warning: you are attempting to run MeshDataModule without the required dependency open3d.")
            raise ModuleNotFoundError()
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        # Ensure path is valid
        root_dir = root_dir.expanduser()
        assert root_dir.exists(), "Path does not exist"
        assert root_dir.is_dir(), "Path is not a directory"

        # Read all indicies
        mesh_ind = self.read_indices(root_dir / 'watertight_meshes.txt')

        if n_total is not None:
            if n_total <= len(mesh_ind):
                mesh_ind = mesh_ind[0:n_total]
            else:
                print(f"Warning: requested n_total ({n_total}) is greater than the number of available meshes ({len(mesh_ind)}). All available meshes will be used.")
        n_total = len(mesh_ind)
        # remove trailing newlines from train and test indices
        mesh_ind = [x.rstrip() for x in mesh_ind]

        data_dir = root_dir / item_dir_name

        # Load all meshes and compute OT
        data = []
        n_non_surjective = 0
        n_missed_points = 0
        for ind in mesh_ind:
            mesh = o3d.io.read_triangle_mesh(
                str(data_dir / ("mesh_" + ind.zfill(3) + ".ply"))
            )
            target = torch.from_numpy(np.asarray(mesh.vertices).squeeze().astype(np.float32())).to(self.device) #(3586,3) car vertices
            mesh.compute_vertex_normals()
            normal = torch.from_numpy(np.asarray(mesh.vertex_normals).astype(np.float32()))
            pressure = np.load(
                str(data_dir / ('press_' + ind.zfill(3) + '.npy'))
            ) 
            pressure = torch.from_numpy(pressure.astype(np.float32()))

            # OT
            n_t = len(target) # number of target samples
            n_s_sqrt = int(np.sqrt(expand_factor)*np.ceil(np.sqrt(n_t))) # sqrt of the number of source samples
            source = self.torus_grid(n_s_sqrt) # build source gird
            _, log = empirical_sinkhorn2_geomloss(X_s=source.to(self.device), X_t=target, reg=reg, log=True) # utilize weighted Sinkhorn a=a.to(device), b=b.to(device),
            gamma = log['lazy_plan'][:].detach() # convert the lazy tensor to torch.tensor (dense)            

            # normalize the OT plan matrix by column
            row_norms = torch.norm(gamma, p=1, dim=1, keepdim=True)
            gamma_encoder = gamma / row_norms

            # transport target to source
            transport = torch.mm(gamma_encoder, target)

            # encoder: target -> source
            distances = torch.cdist(transport, target)
            indices_encoder = torch.argmin(distances, dim=1) # find the closest point in "target" (car vertices) to each point in "transport" (latent grids)
            
            # reset the transport as the closest point in target
            transport = target[indices_encoder]
            unique = len(torch.unique(indices_encoder))
            n_missed_points += n_t - unique
            if unique!=n_t:
                n_non_surjective += 1

            # decoder: source -> target
            indices_decoder = torch.argmin(distances, dim=0) #find the closest point in "transport" (latent grids) to each point in "target" (car vertices)

            item_dict = {
                "target": target.cpu(),
                "source": source.cpu(),
                "ind_enc": indices_encoder.cpu(),
                "ind_dec": indices_decoder.cpu(),
                "press": pressure,
                "nor_t": normal,
                "nor_s": self.torus_normals(n_s_sqrt),
                "trans": transport.cpu()
            }
            data.append(item_dict)
        self.data = data
        torch.save(data, script_dir / "data" / f"ot_expand{expand_factor}_reg{reg}_num{n_total}.pt")
        print(f"Number of none surjective OT plan is {n_non_surjective:d}; average missed points is {n_missed_points/n_total:.2f}")

    def read_indices(self, file_path):
        with open(file_path, 'r') as file:
            indices = [line.strip() for line in file if line.strip().isdigit()]
        return indices
    
    def torus_grid(self, n_s_sqrt, r = 0.5, R = 1.0):
        theta = torch.linspace(0, 2 * torch.pi, n_s_sqrt + 1)[:-1]
        phi = torch.linspace(0, 2 * torch.pi, n_s_sqrt + 1)[:-1]
        # Create a grid using meshgrid
        X, Y = torch.meshgrid(theta, phi, indexing='ij')
        points = torch.stack((X, Y)).reshape((2, -1)).T

        x = (R + r * torch.cos(points[:, 0])) * torch.cos(points[:, 1])
        y = (R + r * torch.cos(points[:, 0])) * torch.sin(points[:, 1])
        z = r * torch.sin(points[:, 0])
        grid = torch.stack((x, y, z), dim=1)
        
        return grid

    def torus_normals(self, n_s_sqrt, r = 0.5, R = 1.0):
        theta = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
        phi = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
        theta, phi = torch.meshgrid(theta, phi, indexing='ij')

        # Partial derivatives
        # dP/dtheta
        dx_dtheta = -r * torch.sin(theta) * torch.cos(phi)
        dy_dtheta = -r * torch.sin(theta) * torch.sin(phi)
        dz_dtheta = r * torch.cos(theta)

        # dP/dphi
        dx_dphi = -(R + r * torch.cos(theta)) * torch.sin(phi)
        dy_dphi = (R + r * torch.cos(theta)) * torch.cos(phi)
        dz_dphi = torch.zeros_like(dx_dphi)

        # Cross product to find normal vectors
        nx = dy_dtheta * dz_dphi - dz_dtheta * dy_dphi
        ny = dz_dtheta * dx_dphi - dx_dtheta * dz_dphi
        nz = dx_dtheta * dy_dphi - dy_dtheta * dx_dphi

        # Stack and normalize
        normals = torch.stack((nx, ny, nz), dim=-1)
        norm = torch.linalg.norm(normals, dim=2, keepdim=True)
        normals = normals / norm

        return normals
