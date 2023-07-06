from typing import List, Tuple, Union, Optional
from collections.abc import Callable
import warnings
import open3d as o3d
import numpy as np
from pathlib import Path
import unittest
import copy

import torch
from torch.utils.data import DataLoader, Dataset

from neuralop.datasets_old.base_datamodule import BaseDataModule
from neuralop.utils import UnitGaussianNormalizer


class DictDataset(Dataset):
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        for k, v in data_dict.items():
            assert len(v) == len(
                data_dict[list(data_dict.keys())[0]]
            ), "All data must have the same length"

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data_dict.items()}

    def __len__(self):
        return len(self.data_dict[list(self.data_dict.keys())[0]])


class DictDatasetWithConstant(DictDataset):
    def __init__(self, data_dict: dict, constant_dict: dict):
        super().__init__(data_dict)
        self.constant_dict = constant_dict

    def __getitem__(self, index):
        return_dict = {k: v[index] for k, v in self.data_dict.items()}
        return_dict.update(self.constant_dict)
        return return_dict


class VariableDictDataset(DictDataset):
    def __init__(
        self,
        data_dict: dict,
        path: str = None,
        location_norm: Optional[Callable] = None,
        pressure_norm: Optional[Callable] = None,
        info_norm: Optional[Callable] = None,
    ):
        DictDataset.__init__(self, data_dict)

        self.path = Path(path) if path is not None else None
        self.location_norm = location_norm
        self.pressure_norm = pressure_norm
        self.info_norm = info_norm

    def index_to_mesh_path(self, index, extension: str = ".ply") -> Path:
        return self.path / ("mesh_" + str(index).zfill(3) + extension)

    def index_to_pressure_path(self, index, extension: str = ".npy") -> Path:
        return self.path / ("press_" + str(index).zfill(3) + extension)
    
    def index_to_info_path(self, index, extension: str = ".pt") -> Path:
        return self.path / ("info_" + str(index).zfill(3) + extension)

    def load_mesh(self, mesh_path: Path) -> o3d.geometry.TriangleMesh:
        assert mesh_path.exists(), "Mesh path does not exist"
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        return mesh

    def load_pressure(self, pressure_path: Path) -> torch.Tensor:
        assert pressure_path.exists(), "Pressure path does not exist"
        pressure = np.load(str(pressure_path)).astype(np.float32)
        return torch.from_numpy(pressure)
    
    def load_info(self, info_path: Path) -> torch.Tensor:
        assert info_path.exists(), "Info path does not exist"
        info = torch.load(str(info_path))
        return info

    def vertices_from_mesh(self, mesh: o3d.geometry.TriangleMesh) -> torch.Tensor:
        return torch.from_numpy(np.asarray(mesh.vertices).astype(np.float32))

    def triangles_from_mesh(self, mesh: o3d.geometry.TriangleMesh) -> torch.Tensor:
        return torch.from_numpy(np.asarray(mesh.triangles).astype(np.int64))

    def get_triangle_centroids(
        self, vertices: torch.Tensor, triangles: torch.Tensor
    ) -> torch.Tensor:
        A, B, C = (
            vertices[triangles[:, 0]],
            vertices[triangles[:, 1]],
            vertices[triangles[:, 2]],
        )

        return (A + B + C) / 3

    def __getitem__(self, index):
        return_dict = {k: v[index] for k, v in self.data_dict.items()}

        if self.path is not None:
            mesh = self.load_mesh(self.index_to_mesh_path(index + 1))
            info = self.load_info(self.index_to_info_path(index + 1))

            vertices = self.vertices_from_mesh(mesh)
            triangles = self.triangles_from_mesh(mesh)
            centroids = self.get_triangle_centroids(vertices, triangles)

            pressure = self.load_pressure(self.index_to_pressure_path(index + 1))

            if self.location_norm is not None:
                vertices = self.location_norm(vertices)
                centroids = self.location_norm(centroids)

            if self.pressure_norm is not None:
                pressure = self.pressure_norm(pressure)
            
            if self.info_norm is not None:
                info = self.info_norm(info)

            return_dict["vertices"] = vertices
            return_dict["centroids"] = centroids
            return_dict["pressure"] = pressure
            return_dict["info"] = info

        return return_dict


class VariableDictDatasetWithConstant(VariableDictDataset):
    def __init__(
        self,
        data_dict: dict,
        constant_dict: dict,
        path: str = None,
        localtion_norm: Optional[Callable] = None,
        pressure_norm: Optional[Callable] = None,
        info_norm: Optional[Callable] = None,
    ):
        super().__init__(data_dict, path, localtion_norm, pressure_norm, info_norm)
        self.constant_dict = constant_dict

    def __getitem__(self, index):
        return_dict = super().__getitem__(index)
        return_dict.update(self.constant_dict)
        return return_dict


class CFDDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir,
        n_train: int = 500,
        n_test: int = 111,
    ):
        super().__init__()

        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        data_dir = data_dir.expanduser()
        assert data_dir.exists(), "Path does not exist"
        assert data_dir.is_dir(), "Path is not a directory"
        self.data_dir = data_dir

        valid_mesh_inds = self.load_valid_mesh_indices(data_dir)
        assert n_train + n_test <= len(valid_mesh_inds), "Not enough data"
        if (n_train + n_test) < len(valid_mesh_inds):
            warnings.warn(
                f"{len(valid_mesh_inds)} meshes are available, but {n_train + n_test} are requested."
            )
        # split the valid_mesh_inds into n_train and n_test indices and use the indices to load the mesh

        train_indices = valid_mesh_inds[:n_train]
        test_indices = valid_mesh_inds[-n_test:]
        train_mesh_pathes = [self.get_mesh_path(data_dir, i) for i in train_indices]
        test_mesh_pathes = [self.get_mesh_path(data_dir, i) for i in test_indices]

        train_vertices = [
            self.vertices_from_mesh(mesh_path) for mesh_path in train_mesh_pathes
        ]
        test_vertices = [
            self.vertices_from_mesh(mesh_path) for mesh_path in test_mesh_pathes
        ]

        train_pressure = torch.stack(
            [
                torch.Tensor(self.load_pressure(data_dir, mesh_index))
                for mesh_index in train_indices
            ]
        )
        test_pressure = torch.stack(
            [
                torch.Tensor(self.load_pressure(data_dir, mesh_index))
                for mesh_index in test_indices
            ]
        )
        # normalize pressure
        pressure_normalization = UnitGaussianNormalizer(
            train_pressure, eps=1e-6, reduce_dim=[0, 1], verbose=False
        )
        train_pressure = pressure_normalization.encode(train_pressure)
        test_pressure = pressure_normalization.encode(test_pressure)

        self._train_data = DictDataset(
            {
                "vertices": train_vertices,
                "pressure": train_pressure,
            },
        )
        self._test_data = DictDataset(
            {
                "vertices": test_vertices,
                "pressure": test_pressure,
            },
        )

        self.output_normalization = pressure_normalization

    def encode(self, pressure: torch.Tensor) -> torch.Tensor:
        self.output_normalization.to(pressure.device)
        return self.output_normalization.encode(pressure)

    def decode(self, pressure: torch.Tensor) -> torch.Tensor:
        self.output_normalization.to(pressure.device)
        return self.output_normalization.decode(pressure)

    @property
    def train_data(self):
        return self._train_data

    @property
    def test_data(self):
        return self._test_data

    def vertices_from_mesh(self, mesh_path: Path) -> torch.Tensor:
        mesh = self.load_mesh(mesh_path)
        vertices = mesh.vertex.positions.numpy()
        return vertices

    def get_mesh_path(self, data_dir: Path, mesh_ind: int) -> Path:
        return data_dir / "data" / ("mesh_" + str(mesh_ind).zfill(3) + ".ply")

    def get_pressure_data_path(self, data_dir: Path, mesh_ind: int) -> Path:
        return data_dir / "data" / ("press_" + str(mesh_ind).zfill(3) + ".npy")

    def load_pressure(self, data_dir: Path, mesh_index: int) -> np.ndarray:
        press_path = self.get_pressure_data_path(data_dir, mesh_index)
        assert press_path.exists(), "Pressure data does not exist"
        press = np.load(press_path).reshape((-1,)).astype(np.float32)
        press = np.concatenate((press[0:16], press[112:]), axis=0)
        return press

    def load_valid_mesh_indices(
        self, data_dir, filename="watertight_meshes.txt"
    ) -> List[int]:
        with open(data_dir / filename, "r") as fp:
            mesh_ind = fp.read().split("\n")
            mesh_ind = [int(a) for a in mesh_ind]
        return mesh_ind

    def load_mesh(self, mesh_path: Path) -> o3d.t.geometry.TriangleMesh:
        assert mesh_path.exists(), "Mesh path does not exist"
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        return mesh

    def load_mesh_from_index(
        self, data_dir, mesh_index: int
    ) -> o3d.t.geometry.TriangleMesh:
        mesh_path = self.get_mesh_path(data_dir, mesh_index)
        return self.load_mesh(mesh_path)


class CFDSDFDataModule(CFDDataModule):
    def __init__(
        self,
        data_dir,
        n_train: int = 500,
        n_test: int = 111,
        spatial_resolution: Tuple[int, int, int] = None,
        query_points=None,
        eps=1e-2,
        closest_points_to_query=True,
    ):
        BaseDataModule.__init__(self)

        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        data_dir = data_dir.expanduser()
        assert data_dir.exists(), "Path does not exist"
        assert data_dir.is_dir(), "Path is not a directory"
        self.data_dir = data_dir

        min_bounds, max_bounds = self.load_bound(data_dir, eps=eps)
        valid_mesh_inds = self.load_valid_mesh_indices(data_dir)
        assert n_train + n_test <= len(valid_mesh_inds), "Not enough data"
        if (n_train + n_test) < len(valid_mesh_inds):
            warnings.warn(
                f"{len(valid_mesh_inds)} meshes are available, but {n_train + n_test} are requested."
            )
        # split the valid_mesh_inds into n_train and n_test indices and use the indices to load the mesh

        train_indices = valid_mesh_inds[:n_train]
        test_indices = valid_mesh_inds[-n_test:]
        train_mesh_pathes = [self.get_mesh_path(data_dir, i) for i in train_indices]
        test_mesh_pathes = [self.get_mesh_path(data_dir, i) for i in test_indices]

        if query_points is None:
            assert spatial_resolution is not None, "spatial_resolution must be given"
            tx = np.linspace(min_bounds[0], max_bounds[0], spatial_resolution[0])
            ty = np.linspace(min_bounds[1], max_bounds[1], spatial_resolution[1])
            tz = np.linspace(min_bounds[2], max_bounds[2], spatial_resolution[2])
            query_points = np.stack(
                np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            ).astype(np.float32)

        train_sdf_mesh_vertices = [
            self.sdf_vertices_closest_from_mesh(
                mesh_path, query_points, closest_points_to_query
            )
            for mesh_path in train_mesh_pathes
        ]
        train_sdf = torch.stack(
            [torch.Tensor(sdf) for sdf, _, _ in train_sdf_mesh_vertices]
        )
        train_vertices = torch.stack(
            [torch.Tensor(vertices) for _, vertices, _ in train_sdf_mesh_vertices]
        )
        if closest_points_to_query:
            train_closest_points = torch.stack(
                [torch.Tensor(closest) for _, _, closest in train_sdf_mesh_vertices]
            )
        else:
            train_closest_points = None

        del train_sdf_mesh_vertices
        train_pressure = torch.stack(
            [
                torch.Tensor(self.load_pressure(data_dir, mesh_index))
                for mesh_index in train_indices
            ]
        )

        # Location normalization
        min_bounds = torch.tensor(min_bounds)
        max_bounds = torch.tensor(max_bounds)

        # normalize train_vertices with min_bounds max_bounds
        train_vertices = self.location_normalization(
            train_vertices, min_bounds, max_bounds
        )

        if closest_points_to_query:
            train_closest_points = self.location_normalization(
                train_closest_points, min_bounds, max_bounds
            ).permute(0, 4, 1, 2, 3)

        test_sdf_mesh_vertices = [
            self.sdf_vertices_closest_from_mesh(
                mesh_path, query_points, closest_points_to_query
            )
            for mesh_path in test_mesh_pathes
        ]
        test_sdf = torch.stack(
            [torch.Tensor(sdf) for sdf, _, _ in test_sdf_mesh_vertices]
        )
        test_vertices = torch.stack(
            [torch.Tensor(vertices) for _, vertices, _ in test_sdf_mesh_vertices]
        )
        if closest_points_to_query:
            test_closest_points = torch.stack(
                [torch.Tensor(closest) for _, _, closest in test_sdf_mesh_vertices]
            )
        else:
            test_closest_points = None

        del test_sdf_mesh_vertices
        test_pressure = torch.stack(
            [
                torch.Tensor(self.load_pressure(data_dir, mesh_index))
                for mesh_index in test_indices
            ]
        )
        test_vertices = self.location_normalization(
            test_vertices, min_bounds, max_bounds
        )
        if closest_points_to_query:
            test_closest_points = self.location_normalization(
                test_closest_points, min_bounds, max_bounds
            ).permute(0, 4, 1, 2, 3)

        # normalize pressure
        pressure_normalization = UnitGaussianNormalizer(
            train_pressure, eps=1e-6, reduce_dim=[0, 1], verbose=False
        )
        train_pressure = pressure_normalization.encode(train_pressure)
        test_pressure = pressure_normalization.encode(test_pressure)

        # normalize query points
        normalized_query_points = self.location_normalization(
            torch.from_numpy(query_points), min_bounds, max_bounds
        ).permute(3, 0, 1, 2)

        self._train_data = DictDatasetWithConstant(
            {
                "sdf": train_sdf,
                "vertices": train_vertices,
                "pressure": train_pressure,
            },
            {"sdf_query_points": normalized_query_points},
        )
        self._test_data = DictDatasetWithConstant(
            {
                "sdf": test_sdf,
                "vertices": test_vertices,
                "pressure": test_pressure,
            },
            {"sdf_query_points": normalized_query_points},
        )

        if closest_points_to_query:
            self._train_data.data_dict["closest_points"] = train_closest_points
            self._test_data.data_dict["closest_points"] = test_closest_points

        self.output_normalization = pressure_normalization

    def load_bound(
        self, data_dir, filename="watertight_global_bounds.txt", eps=1e-6
    ) -> Tuple[List[float], List[float]]:
        with open(data_dir / filename, "r") as fp:
            min_bounds = fp.readline().split(" ")
            max_bounds = fp.readline().split(" ")

            min_bounds = [float(a) - eps for a in min_bounds]
            max_bounds = [float(a) + eps for a in max_bounds]
        return min_bounds, max_bounds

    def location_normalization(
        self,
        locations: torch.Tensor,
        min_bounds: torch.Tensor,
        max_bounds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize locations to [-1, 1].
        """
        locations = (locations - min_bounds) / (max_bounds - min_bounds)
        locations = 2 * locations - 1
        return locations

    def compute_sdf(
        self, mesh: Union[Path, o3d.t.geometry.TriangleMesh], query_points
    ) -> np.ndarray:
        if isinstance(mesh, Path):
            mesh = self.load_mesh(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        signed_distance = scene.compute_signed_distance(query_points).numpy()
        return signed_distance

    def closest_points_to_query_from_mesh(
        self, mesh: o3d.t.geometry.TriangleMesh, query_points
    ) -> np.ndarray:
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)

        closest_points = scene.compute_closest_points(query_points)["points"].numpy()

        return closest_points

    def sdf_vertices_closest_from_mesh(
        self,
        mesh_path: Path,
        query_points: np.ndarray,
        closest_points: bool,
    ) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        mesh = self.load_mesh(mesh_path)
        sdf = self.compute_sdf(mesh, query_points)
        vertices = mesh.vertex.positions.numpy()
        if closest_points:
            closest_points = self.closest_points_to_query_from_mesh(mesh, query_points)
        else:
            closest_points = None

        return sdf, vertices, closest_points


class AhmedBodyDataModule(CFDSDFDataModule):
    def __init__(
        self,
        data_dir,
        n_train: int = 500,
        n_test: int = 51,
        spatial_resolution: Tuple[int, int, int] = None,
        query_points=None,
        eps=1e-2,
        closest_points_to_query=True,
    ):
        BaseDataModule.__init__(self)

        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        data_dir = data_dir.expanduser()
        assert data_dir.exists(), "Path does not exist"
        assert data_dir.is_dir(), "Path is not a directory"
        self.data_dir = data_dir

        min_bounds, max_bounds = self.load_bound(
            data_dir, filename="global_bounds.txt", eps=eps
        )

        min_info_bounds, max_info_bounds = self.load_bound(
            data_dir, filename='info_bounds.txt', eps=0.0
        )

        assert n_train <= 500, "Not enough training data"
        assert n_test <= 51, "Not enough testing data"
        if (n_train + n_test) < 551:
            warnings.warn(
                f"551 meshes are available, but {n_train + n_test} are requested."
            )

        train_indices = [j + 1 for j in range(n_train)]
        test_indices = [j + 1 for j in range(n_test)]
        train_mesh_pathes = [
            self.get_mesh_path(data_dir, "train", i) for i in train_indices
        ]
        test_mesh_pathes = [
            self.get_mesh_path(data_dir, "test", i) for i in test_indices
        ]

        if query_points is None:
            assert spatial_resolution is not None, "spatial_resolution must be given"
            tx = np.linspace(min_bounds[0], max_bounds[0], spatial_resolution[0])
            ty = np.linspace(min_bounds[1], max_bounds[1], spatial_resolution[1])
            tz = np.linspace(min_bounds[2], max_bounds[2], spatial_resolution[2])
            query_points = np.stack(
                np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            ).astype(np.float32)

        train_df_closest = [
            self.df_from_mesh(mesh_path, query_points, closest_points_to_query)
            for mesh_path in train_mesh_pathes
        ]
        train_df = torch.stack([torch.Tensor(df) for df, _ in train_df_closest])
        if closest_points_to_query:
            train_closest_points = torch.stack(
                [torch.Tensor(closest) for _, closest in train_df_closest]
            )
        else:
            train_closest_points = None

        del train_df_closest

        test_df_closest = [
            self.df_from_mesh(mesh_path, query_points, closest_points_to_query)
            for mesh_path in test_mesh_pathes
        ]
        test_df = torch.stack([torch.Tensor(df) for df, _ in test_df_closest])
        if closest_points_to_query:
            test_closest_points = torch.stack(
                [torch.Tensor(closest) for _, closest in test_df_closest]
            )
        else:
            test_closest_points = None

        del test_df_closest

        # normalize pressure
        train_pressure = torch.cat(
            [
                torch.from_numpy(self.load_pressure(data_dir, "train", i))
                for i in train_indices
            ]
        )

        pressure_normalization = UnitGaussianNormalizer(
            train_pressure, eps=1e-6, reduce_dim=[0], verbose=False
        )

        min_bounds = torch.tensor(min_bounds)
        max_bounds = torch.tensor(max_bounds)

        normalized_query_points = self.location_normalization(
            torch.from_numpy(query_points), min_bounds, max_bounds
        ).permute(3, 0, 1, 2)

        if closest_points_to_query:
            train_closest_points = self.location_normalization(
                train_closest_points, min_bounds, max_bounds
            ).permute(0, 4, 1, 2, 3)

            test_closest_points = self.location_normalization(
                test_closest_points, min_bounds, max_bounds
            ).permute(0, 4, 1, 2, 3)

        location_norm_fn = lambda x: self.location_normalization(
            x, min_bounds, max_bounds
        )
        info_norm_fn = lambda x: self.info_normalization(
            x, min_info_bounds, max_info_bounds
        )
        self._train_data = VariableDictDatasetWithConstant(
            {
                "df": train_df,
            },
            {"df_query_points": normalized_query_points},
            self.data_dir / "train",
            localtion_norm=location_norm_fn,
            pressure_norm=copy.deepcopy(pressure_normalization).encode,
            info_norm=info_norm_fn
        )
        self._test_data = VariableDictDatasetWithConstant(
            {
                "df": test_df,
            },
            {"df_query_points": normalized_query_points},
            self.data_dir / "test",
            localtion_norm=location_norm_fn,
            pressure_norm=copy.deepcopy(pressure_normalization).encode,
            info_norm=info_norm_fn
        )

        if closest_points_to_query:
            self._train_data.data_dict["closest_points"] = train_closest_points
            self._test_data.data_dict["closest_points"] = test_closest_points

        self._aggregatable = list(self._train_data.data_dict.keys()) + list(
            self._train_data.constant_dict.keys()
        )

        self.output_normalization = pressure_normalization

    def get_mesh_path(self, data_dir: Path, subfolder: str, mesh_ind: int) -> Path:
        return data_dir / subfolder / ("mesh_" + str(mesh_ind).zfill(3) + ".ply")

    def get_pressure_data_path(
        self, data_dir: Path, subfolder: str, mesh_ind: int
    ) -> Path:
        return data_dir / subfolder / ("press_" + str(mesh_ind).zfill(3) + ".npy")

    def load_pressure(
        self, data_dir: Path, subfolder: str, mesh_index: int
    ) -> np.ndarray:
        press_path = self.get_pressure_data_path(data_dir, subfolder, mesh_index)
        assert press_path.exists(), "Pressure data does not exist"
        press = np.load(press_path).reshape((-1,)).astype(np.float32)
        return press

    def compute_df(
        self, mesh: Union[Path, o3d.t.geometry.TriangleMesh], query_points
    ) -> np.ndarray:
        if isinstance(mesh, Path):
            mesh = self.load_mesh(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        distance = scene.compute_distance(query_points).numpy()
        return distance

    def df_from_mesh(
        self,
        mesh_path: Path,
        query_points: np.ndarray,
        closest_points: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mesh = self.load_mesh(mesh_path)
        df = self.compute_df(mesh, query_points)
        if closest_points:
            closest_points = self.closest_points_to_query_from_mesh(mesh, query_points)
        else:
            closest_points = None

        return df, closest_points

    def info_normalization(
        self,
        info: dict,
        min_bounds: List[float],
        max_bounds: List[float],
    ) -> dict:
        """
        Normalize info to [0, 1].
        """
        for i, (k, v) in enumerate(info.items()):
            info[k] = (v - min_bounds[i]) / (max_bounds[i] - min_bounds[i])

        return info

    def collate_fn(self, batch):
        aggr_dict = {}
        for key in self._aggregatable:
            aggr_dict.update(
                {key: torch.stack([data_dict[key] for data_dict in batch])}
            )

        remaining = list(set(batch[0].keys()) - set(self._aggregatable))
        for key in remaining:
            aggr_dict.update({key: [data_dict[key] for data_dict in batch]})

        return aggr_dict


class TestCFD(unittest.TestCase):
    def __init__(self, methodName: str, data_path: str) -> None:
        super().__init__(methodName)
        self.data_path = data_path

    def test_cfd(self):
        dm = CFDDataModule(
            self.data_path,
            n_train=10,
            n_test=10,
        )
        tl = dm.train_dataloader(batch_size=2, shuffle=True)
        for batch in tl:
            for k, v in batch.items():
                print(k, v.shape)
            break

    def test_cfd_grid(self):
        dm = CFDSDFDataModule(
            self.data_path,
            n_train=10,
            n_test=10,
            spatial_resolution=(64, 64, 64),
        )
        tl = dm.train_dataloader(batch_size=2, shuffle=True)
        for batch in tl:
            for k, v in batch.items():
                print(k, v.shape)
            break


class TestAhmed(unittest.TestCase):
    def __init__(self, methodName: str, data_path: str) -> None:
        super().__init__(methodName)
        self.data_path = data_path

    def test_ahmed(self):
        dm = AhmedBodyDataModule(
            self.data_path,
            n_train=10,
            n_test=10,
            spatial_resolution=(64, 64, 64),
        )

        tl = dm.train_dataloader(batch_size=2, shuffle=True)
        for batch in tl:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(k, v.shape)
                else:
                    print(k)
                    for j in range(len(v)):
                        if isinstance(v[j], dict):
                            print(v[j])
                        else:
                            print(v[j].shape)
            break


if __name__ == "__main__":
    data_dir_cfd = Path("~/datasets/geono/car-pressure-data").expanduser()
    data_dir_ahmed = Path("~/datasets/geono/ahmed").expanduser()

    # Unittest
    test_suite = unittest.TestSuite()
    # TestCFD and setup the path
    test_suite.addTest(TestCFD("test_cfd", data_dir_cfd))
    test_suite.addTest(TestCFD("test_cfd_grid", data_dir_cfd))
    test_suite.addTest(TestAhmed("test_ahmed", data_dir_ahmed))
    unittest.TextTestRunner().run(test_suite)
