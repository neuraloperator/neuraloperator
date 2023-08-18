from typing import List, Tuple, Union, Optional, Literal
import sys
from collections.abc import Callable
import warnings
import open3d as o3d
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from neuralop.utils import UnitGaussianNormalizer


class CFDDataset(Dataset):
    def __init__(
        self,
        data_dir,
        phase: Literal["train", "test"] = "train",
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

        if phase == "train":
            mesh_indices = valid_mesh_inds[:n_train]
        elif phase == "test":
            mesh_indices = valid_mesh_inds[-n_test:]
        else:
            raise ValueError("Invalid phase")

        mesh_pathes = [self.get_mesh_path(data_dir, i) for i in mesh_indices]
        vertices = [self.vertices_from_mesh(mesh_path) for mesh_path in mesh_pathes]
        pressures = torch.stack(
            [
                torch.Tensor(self.load_pressure(data_dir, mesh_index))
                for mesh_index in mesh_indices
            ]
        )
        self.vertices = vertices
        self.pressures = pressures

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

    def encode(self, pressure_normalization: UnitGaussianNormalizer):
        self.pressures = pressure_normalization.encode(self.pressures)

    def __len__(self):
        return len(self.vertices)

    def __getitem__(self, index):
        return self.vertices[index], self.pressures[index]


def load_cfd(data_path, n_train, n_test, batch_size, test_batch_size):
    # normalize pressure
    train_dataset = CFDDataset(data_path, phase="train", n_train=n_train, n_test=n_test)
    test_dataset = CFDDataset(data_path, phase="test", n_train=n_train, n_test=n_test)

    pressure_normalization = UnitGaussianNormalizer(
        train_dataset.pressures, eps=1e-6, reduce_dim=[0, 1], verbose=False
    )
    train_dataset.encode(pressure_normalization)
    test_dataset.encode(pressure_normalization)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    return train_loader, test_loader, pressure_normalization
