"""
Tests fallback neighbor search on a small 2d grid
that was calculated manually
"""

import numpy as np
import torch
import pytest

from ..neighbor_search import native_neighbor_search

# Manually-calculated CSR list of neighbors
# in a 5x5 grid on [0,1] X [0,1] for radius=0.3

indices = [0, 1, 5, 0, 1, 2, 6, 1, 2, 3, 7, 2, 3, 4, 8,
           3, 4, 9, 0, 5, 6, 10, 1, 5, 6, 7, 11, 2, 6, 7,
           8, 12, 3, 7, 8, 9, 13, 4, 8, 9, 14, 5, 10, 11,
           15, 6, 10, 11, 12, 16, 7, 11, 12, 13, 17, 8, 12,
           13, 14, 18, 9, 13, 14, 19, 10, 15, 16, 20, 11, 15,
           16, 17, 21, 12, 16, 17, 18, 22, 13, 17, 18, 19, 23,
           14, 18, 19, 24, 15, 20, 21, 16, 20, 21, 22, 17, 21,
           22, 23, 18, 22, 23, 24, 19, 23, 24]

splits = [0, 3, 7, 11, 15, 18, 22, 27, 32, 37, 41, 45, 50,
          55, 60, 64, 68, 73, 78, 83, 87, 90, 94, 98, 102, 105]

def test_fallback_nb_search():
    mesh_grid = np.stack(
        np.meshgrid(*[np.linspace(0, 1, 5) for _ in range(2)], indexing="ij"), axis=-1
    )
    coords = torch.Tensor(
        mesh_grid.reshape(-1, 2)
    )  # reshape into n**d x d coord points
    return_dict = native_neighbor_search(
        data=coords, queries=coords, radius=0.3, return_norm=True
    )

    assert return_dict["neighbors_index"].tolist() == indices
    assert return_dict["neighbors_row_splits"].tolist() == splits
    print(f"{return_dict['weights']=}")

    def compute_norm_separate(nbrs, data, queries):
        return_dict = nbrs
        num_reps = (
            return_dict["neighbors_row_splits"][1:]
            - return_dict["neighbors_row_splits"][:-1]
        )
        rep_queries = torch.repeat_interleave(queries, num_reps, dim=0)
        rep_data = data[return_dict["neighbors_index"]]
        rep_dist = rep_queries - rep_data
        return_dict["squared_norm"] = (rep_dist**2).sum(dim=-1)
        return return_dict

    return_dict = compute_norm_separate(return_dict, coords, coords)
    print(return_dict["squared_norm"])


def test_large_neighbor_counts_keep_csr_consistent():
    # This creates just over 2**24 coincident pairs, the point where float32
    # cumulative counts can no longer represent every integer exactly.
    n_points = 4097
    data = torch.zeros(n_points, 1)
    queries = torch.zeros(n_points, 1)

    result = native_neighbor_search(data, queries, radius=1.0)
    expected_pairs = n_points * n_points
    expected_splits = torch.arange(
        0, expected_pairs + 1, n_points, dtype=torch.long
    )

    assert result["neighbors_index"].numel() == expected_pairs
    assert result["neighbors_row_splits"].dtype == torch.long
    torch.testing.assert_close(result["neighbors_row_splits"], expected_splits)
