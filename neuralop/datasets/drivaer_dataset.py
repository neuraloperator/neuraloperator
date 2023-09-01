from typing import Union, Literal, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import zarr

try:
    import ensightreader
except ImportError:
    print(
        "Could not import ensightreader. Please install it from `pip install ensight-reader`"
    )


class DrivAerDataset(Dataset):
    """DrivAer dataset"""

    def __init__(
        self,
        data_path: Union[str, Path],
        phase: Literal["train", "val", "test"] = "train",
        has_spoiler: bool = False,
        variables: Optional[list] = ["time_avg_pressure", "time_avg_wall_shear_stress"],
    ):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        self.data_path = data_path  # Path that contains data_set_A and data_set_B
        assert self.data_path.exists(), f"Path {self.data_path} does not exist"
        assert self.data_path.is_dir(), f"Path {self.data_path} is not a directory"
        assert (
            self.data_path / "data_set_A"
        ).exists(), f"Path {self.data_path} does not contain data_set_A"

        if has_spoiler:
            self.data_path = self.data_path / "data_set_B"
            self.TEST_INDS = np.array(range(510, 600))
            self.VAL_INDS = np.array(range(420, 510))
            self.TRAIN_INDS = np.array(range(420))
        else:
            self.data_path = self.data_path / "data_set_A"
            self.TEST_INDS = np.array(range(340, 400))
            self.VAL_INDS = np.array(range(280, 340))
            self.TRAIN_INDS = np.array(range(280))

        # Load parameters
        parameters = pd.read_csv(
            self.data_path / "ParameterFile.txt", delim_whitespace=True
        )
        if phase == "train":
            self.indices = self.TRAIN_INDS
        elif phase == "val":
            self.indices = self.VAL_INDS
        elif phase == "test":
            self.indices = self.TEST_INDS
        self.parameters = parameters.iloc[self.indices]
        self.variables = variables

    @property
    def _attribute(self, variable, name):
        return self.data[variable].attrs[name]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        case = ensightreader.read_case(
            self.data_path / "snapshots" / f"EnSight{index}" / f"EnSight{index}.case"
        )
        geofile = case.get_geometry_model()
        ids = geofile.get_part_ids()  # list of part ids
        # remove id 49, which is the internalMesh
        ids.remove(49)

        coordinate_list = []
        center_coordinate_list = []
        data_list = []
        for part_id in ids:
            part = geofile.get_part_by_id(part_id)
            element_blocks = part.element_blocks
            with geofile.open() as fp_geo:
                part_coordinates = part.read_nodes(fp_geo)
            coordinate_list.append(part_coordinates)

            # Read element data
            cell_center_coordinate_list = []
            with geofile.open() as fp_geo:
                for block in part.element_blocks:
                    if block.element_type == ensightreader.ElementType.NFACED:
                        (
                            polyhedra_face_counts,
                            face_node_counts,
                            face_connectivity,
                        ) = block.read_connectivity_nfaced(fp_geo)
                        cell_center_coordinates = compute_avg_coordinates_polyhedra(
                            polyhedra_face_counts,
                            face_node_counts,
                            face_connectivity - 1,
                            part_coordinates,
                        )
                    elif block.element_type == ensightreader.ElementType.NSIDED:
                        (
                            polygon_node_counts,
                            polygon_connectivity,
                        ) = block.read_connectivity_nsided(fp_geo)
                        # index start from 1
                        cell_center_coordinates = compute_avg_coordinates(
                            polygon_connectivity - 1,
                            polygon_node_counts,
                            part_coordinates,
                        )
                    else:
                        connectivity = block.read_connectivity(fp_geo)
                        cell_coordinates = part_coordinates[
                            connectivity - 1
                        ]  # N x F x 3
                        cell_center_coordinates = np.average(cell_coordinates, axis=1)
                    cell_center_coordinate_list.append(cell_center_coordinates)
            cell_center_coordinates = np.concatenate(cell_center_coordinate_list)
            center_coordinate_list.append(cell_center_coordinates)

            # Get variables
            out_variables = []
            for variable_name in self.variables:
                variable = case.get_variable(variable_name)
                blocks = []
                for element_block in element_blocks:
                    with variable.mmap() as mm_var:
                        data = variable.read_element_data(
                            mm_var, part.part_id, element_block.element_type
                        )
                        blocks.append(data)
                    data = np.concatenate(blocks)
                # scalar variables are transformed to N,1 arrays
                if len(data.shape) == 1:
                    data = data[:, np.newaxis]
                out_variables.append(data)
            data_list.append(np.concatenate(out_variables, axis=-1))

        coordinates = np.concatenate(coordinate_list)
        center_coordinates = np.concatenate(center_coordinate_list)
        data = np.concatenate(data_list)
        return {
            "mesh_nodes": coordinates,
            "cell_centers": center_coordinates,
            "data": data,
        }


def compute_avg_coordinates(connectivity, polygon_node_counts, coordinates):
    # Create an array of starting indices for each polygon
    start_indices = np.insert(np.cumsum(polygon_node_counts)[:-1], 0, 0)

    # Compute average coordinates for each polygon
    avg_coords = []
    for start, count in zip(start_indices, polygon_node_counts):
        # Extract node IDs of the current polygon
        polygon_nodes = connectivity[start : start + count]

        # Calculate the average coordinates
        avg = np.mean(coordinates[polygon_nodes], axis=0)
        avg_coords.append(avg)

    return np.array(avg_coords)


def compute_avg_coordinates_polyhedra(
    polyhedra_face_counts, face_node_counts, face_connectivity, coordinates
):
    # Create an array of starting indices for each polyhedron's faces and for each face's nodes
    polyhedra_start_indices = np.insert(np.cumsum(polyhedra_face_counts)[:-1], 0, 0)
    face_start_indices = np.insert(np.cumsum(face_node_counts)[:-1], 0, 0)

    avg_coords = []
    face_idx = 0  # keep track of current face in face_connectivity

    for polyhedra_start, polyhedra_count in zip(
        polyhedra_start_indices, polyhedra_face_counts
    ):
        polyhedra_nodes = []
        for _ in range(polyhedra_count):
            face_nodes = face_connectivity[
                face_start_indices[face_idx] : face_start_indices[face_idx]
                + face_node_counts[face_idx]
            ]
            polyhedra_nodes.extend(face_nodes)
            face_idx += 1

        # Convert to numpy array and then compute the average coordinates for the polyhedron
        polyhedra_nodes = np.unique(np.array(polyhedra_nodes))  # remove duplicates
        avg = np.mean(coordinates[polyhedra_nodes], axis=0)
        avg_coords.append(avg)

    return np.array(avg_coords)


class DrivAerToZarr:
    def __init__(self, dataset: DrivAerDataset, output_path: Union[str, Path]):
        self.dataset = dataset
        self.output_path = Path(output_path)

    def save(self):
        # Create a zarr directory store
        store = zarr.DirectoryStore(str(self.output_path))
        root = zarr.group(store=store, overwrite=True)

        # Initialize arrays to hold data
        mesh_nodes_array = root.zeros(
            "mesh_nodes", shape=(0, 3), chunks=(100, 3), dtype="float32"
        )
        cell_centers_array = root.zeros(
            "cell_centers", shape=(0, 3), chunks=(100, 3), dtype="float32"
        )
        data_array = root.zeros(
            "data",
            shape=(0, self.dataset.variables.shape[-1]),
            chunks=(100, len(self.dataset.variables)),
            dtype="float32",
        )

        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            mesh_nodes_array.append(item["mesh_nodes"])
            cell_centers_array.append(item["cell_centers"])
            data_array.append(item["data"])

        # Optionally, you can compress the data if needed using zarr's built-in compression,
        # but for simplicity, this has been omitted in the above example.


if __name__ == "__main__":
    # Get the path from argv
    import sys

    data_path = sys.argv[-1]
    dataset = DrivAerDataset(data_path, phase="train", has_spoiler=False)
    print(dataset[0])
