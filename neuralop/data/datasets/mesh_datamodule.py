from pathlib import Path
from timeit import default_timer
from typing import List, Union

import numpy as np

# import open3d for io if built. Otherwise,
# the class will build, but no files will be loaded.
try:
    import open3d as o3d
except ModuleNotFoundError:
    print("Warning: you are attempting to run MeshDataModule without the required dependency open3d.")

import torch
from torch.utils.data import DataLoader

from .dict_dataset import DictDataset
from ..transforms.normalizers import UnitGaussianNormalizer


class MeshDataModule:
    def __init__(
        self,
        root_dir: Union[str, Path],
        item_dir_name: Union[str, Path],
        n_train: int = None,
        n_test: int = None,
        query_res: List[int] = None,
        attributes: List[str] = None,
    ):
        """MeshDataModule provides a general dataset for irregular coordinate meshes
            for use in a GNO-based architecture

        Parameters
        ----------
        root_dir : Union[str, Path]
            str or Path to root directory of CFD dataset
        item_dir_name : Union[str, Path]
            directory in which individual item subdirs are stored
        n_train : int, optional
            hard limit on number of training examples
            if n_train is greater than the actual number
            of training examples available, nothing is changed
        n_test : int, optional
            hard limit on number of test examples
            if n_test is greater than the actual number
            of testing examples available, nothing is changed
        query_res : List[int], optional
            resolution of latent query points along each dimension
        attributes : List[str], optional
            list of string keys for attributes in the dataset to return
            as keys for each batch dict
        """

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        # Ensure path is valid
        root_dir = root_dir.expanduser()
        assert root_dir.exists(), "Path does not exist"
        assert root_dir.is_dir(), "Path is not a directory"

        # Read train and test indicies
        with open(root_dir / "train.txt") as file:
            train_ind = file.readline().split(",")

        with open(root_dir / "test.txt") as file:
            test_ind = file.readline().split(",")

        if n_train is not None:
            if n_train < len(train_ind):
                train_ind = train_ind[0:n_train]

        if n_test is not None:
            if n_test < len(test_ind):
                test_ind = test_ind[0:n_test]

        # set train and test sizes
        train_ind = train_ind[0:n_train]
        test_ind = test_ind[0:n_test]
        n_train = len(train_ind)
        n_test = len(test_ind)
        print("n_train n_test are", n_train, n_test)

        mesh_ind = train_ind + test_ind
        # remove trailing newlines from train and test indices
        mesh_ind = [x.rstrip() for x in mesh_ind]

        data_dir = root_dir / "data"

        # Load all meshes

        meshes = []
        for ind in mesh_ind:
            mesh = o3d.io.read_triangle_mesh(
                str(data_dir / (item_dir_name + ind + "/tri_mesh.ply"))
            )
            meshes.append(mesh)

        # Dataset wide bounding box
        min_b, max_b = self.get_global_bounding_box(meshes)

        # are_watertight = self.are_watertight(meshes)
        are_watertight = True

        # Uniform query points if not provided
        if isinstance(query_res, list) or isinstance(query_res, tuple):
            tx = np.linspace(min_b[0], max_b[0], query_res[0])
            ty = np.linspace(min_b[1], max_b[1], query_res[1])
            tz = np.linspace(min_b[2], max_b[2], query_res[2])

            query_points = np.stack(
                np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            ).astype(np.float32)
        else:
            raise TypeError()

        # Compute data from meshes
        data = []
        deleted_meshes = []
        self.time_to_distance = 0.0
        for i, mesh in enumerate(meshes):
            item_dict = {}

            mesh = mesh.compute_triangle_normals()
            mesh = mesh.compute_vertex_normals()

            item_dict["vertices"] = np.asarray(mesh.vertices)
            item_dict["vertex_normals"] = np.asarray(mesh.vertex_normals)
            item_dict["triangle_normals"] = np.asarray(mesh.triangle_normals)

            centroids, area = self.compute_triangle_centroids(
                item_dict["vertices"], np.asarray(mesh.triangles)
            )

            item_dict["centroids"] = centroids
            item_dict["triangle_areas"] = area

            # Normalize vertex data based on global bound
            item_dict["vertices"] = self.range_normalize(
                item_dict["vertices"], min_b, max_b, 0, 1
            )

            item_dict["centroids"] = self.range_normalize(
                item_dict["centroids"], min_b, max_b, 0, 1
            )

            if query_points is not None:
                tt = default_timer()
                try:
                    distance, closest = self.compute_distances(
                        mesh, query_points, are_watertight
                    )
                except:
                    deleted_meshes.append(mesh_ind[i])
                    print(f"{i}-th mesh is empty and will not be added to the dataset")
                    continue
                self.time_to_distance += default_timer() - tt
                item_dict["distance"] = np.expand_dims(distance, -1)
                item_dict["closest_points"] = closest

                # Normalize vertex data based on global bound
                item_dict["closest_points"] = self.range_normalize(
                    item_dict["closest_points"], min_b, max_b, 0, 1
                )
            data.append(item_dict)

        self.time_to_distance /= len(meshes)

        del meshes

        # remove all broken meshes from training set
        n_train -= len(deleted_meshes)
        print(f"{deleted_meshes=}")

        # Bounds based on training data
        min_dist, max_dist = self.get_bounds_from_data(data[0:n_train], "distance")
        min_area, max_area = self.get_bounds_from_data(
            data[0:n_train], "triangle_areas"
        )

        for data_dict in data:
            data_dict["distance"] = self.range_normalize(
                data_dict["distance"], min_dist, max_dist, 1e-6, 1
            )
            data_dict["normalized_triangle_areas"] = self.range_normalize(
                data_dict["triangle_areas"], min_area, max_area, 1e-6, 1
            )

        # Convert to torch
        for data_dict in data:
            for key in data_dict:
                data_dict[key] = torch.from_numpy(data_dict[key]).to(torch.float32)

        # Load non-mesh data
        if attributes is not None:
            for j, ind in enumerate(mesh_ind):
                # skip corrupted meshes we caught while adding to dataset
                if ind in deleted_meshes:
                    print(f"{j}-th pressure field ind {ind} was deleted.")
                    continue
                for attr in attributes:
                    path = str(data_dir / (item_dir_name + ind + "/" + attr + ".npy"))
                    data[j][attr] = torch.from_numpy((np.load(path)))

                    if isinstance(data[j][attr], torch.Tensor):
                        data[j][attr] = data[j][attr].to(torch.float32)

            # Compute Gaussian normalizers based on training data
            normalizer_keys = []
            for attr in attributes:
                if isinstance(data[0][attr], torch.Tensor):
                    normalizer_keys.append(attr)
            # returns keyed dict of UnitGaussianNormalizer instances
            self.normalizers = UnitGaussianNormalizer.from_dataset(
                data, dim=[1], keys=normalizer_keys
            )

            # Encode all data
            for attr in normalizer_keys:
                for j in range(len(data)):
                    data_elem = data[j][attr]
                    if data_elem.shape[0] != 1:
                        data_elem = data_elem.unsqueeze(0)
                    data[j][attr] = self.normalizers[attr].transform(data_elem)

            if not bool(self.normalizers):
                self.normalizers = None
        else:
            self.normalizers = None

        # Set-up constant dict
        query_points = self.range_normalize(query_points, min_b, max_b, 0, 1)

        query_points = torch.from_numpy(query_points).to(torch.float32)
        constant = {"query_points": query_points}

        # Datasets
        self.train_data = DictDataset(data[0:n_train], constant)
        self.test_data = DictDataset(data[n_train:], constant)

    def get_global_bounding_box(self, meshes):
        min_b = np.zeros((3, len(meshes)))
        max_b = np.zeros((3, len(meshes)))
        for j, mesh in enumerate(meshes):
            try:
                min_b[:, j] = mesh.get_min_bound()
                max_b[:, j] = mesh.get_max_bound()
            except IndexError:
                print(f"{j}-th mesh could not be bounded. ")
                pass

        min_b = min_b.min(axis=1)
        max_b = max_b.max(axis=1)

        return min_b, max_b

    def are_watertight(self, meshes):
        for mesh in meshes:
            if not mesh.is_watertight():
                return False

        return True

    def compute_triangle_centroids(self, vertices, triangles):
        A, B, C = (
            vertices[triangles[:, 0]],
            vertices[triangles[:, 1]],
            vertices[triangles[:, 2]],
        )

        centroids = (A + B + C) / 3
        areas = np.sqrt(np.sum(np.cross(B - A, C - A) ** 2, 1)) / 2

        return centroids, areas

    def compute_distances(self, mesh, query_points, signed_distance):
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)

        if signed_distance:
            dist = scene.compute_signed_distance(query_points).numpy()
        else:
            dist = scene.compute_distance(query_points).numpy()

        closest = scene.compute_closest_points(query_points)["points"].numpy()

        return dist, closest

    def range_normalize(self, data, min_b, max_b, new_min, new_max):
        data = (data - min_b) / (max_b - min_b)
        data = (new_max - new_min) * data + new_min

        return data

    def get_bounds_from_data(self, data, key):
        global_min = data[0][key].min()
        global_max = data[0][key].max()

        for j in range(1, len(data)):
            current_min = data[j][key].min()
            current_max = data[j][key].max()

            if current_min < global_min:
                global_min = current_min
            if current_max > global_max:
                global_max = current_max

        return global_min, global_max

    def train_loader(self, **kwargs):
        return DataLoader(self.train_data, **kwargs)

    def test_loader(self, **kwargs):
        return DataLoader(self.test_data, **kwargs)
