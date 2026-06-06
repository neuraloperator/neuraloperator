from ..burgers import Burgers1dTimeDataset
from ..darcy import DarcyDataset
from ..navier_stokes import NavierStokesDataset
from ..pt_dataset import PTDataset
from pathlib import Path

import os
import shutil
import tempfile

import torch
import pytest

test_data_dir = Path("./dataset_test")


@pytest.mark.parametrize("resolution", [16])
def test_DarcyDatasetDownload(resolution):
    dataset = DarcyDataset(
        root_dir=test_data_dir,
        n_train=5,
        n_tests=[5],
        batch_size=1,
        test_batch_sizes=[1],
        train_resolution=resolution,
        test_resolutions=[resolution, resolution * 2],
        download=True,
    )

    downloaded_files = os.listdir(test_data_dir)
    for split in ["train", "test"]:
        for res in [resolution, resolution * 2]:
            assert f"darcy_{split}_{res}.pt" in downloaded_files
    assert dataset.train_db
    assert dataset.test_dbs
    assert dataset.data_processor
    shutil.rmtree(test_data_dir)


burgers_root = Path("./neuralop/data/datasets/data/").resolve()


@pytest.mark.parametrize("resolution", [16])
def test_BurgersDataset(resolution):
    dataset = Burgers1dTimeDataset(
        root_dir=burgers_root,
        n_train=5,
        n_tests=[5],
        batch_size=1,
        test_batch_sizes=[1],
        train_resolution=resolution,
        test_resolutions=[resolution],
    )

    assert dataset.train_db
    assert dataset.test_dbs
    assert dataset.data_processor


def test_NSDownload():
    # monkeypatch bypasses confirmation input
    dataset = NavierStokesDataset(
        root_dir=test_data_dir,
        n_train=5,
        n_tests=[5],
        batch_size=1,
        test_batch_sizes=[1],
        train_resolution=128,
        test_resolutions=[128],
        download=True,
    )

    downloaded_files = os.listdir(test_data_dir)
    for split in ["train", "test"]:
        for res in [128]:
            assert f"nsforcing_{split}_{res}.pt" in downloaded_files
    assert dataset.train_db
    assert dataset.test_dbs
    assert dataset.data_processor
    shutil.rmtree(test_data_dir)


# --- dtype parameter tests ---

def _write_mock_pt_files(directory, x_dtype, n=10, length=8):
    """Write minimal train/test .pt files with x tensors of the given dtype."""
    data = {
        "x": torch.ones(n, length, dtype=x_dtype),
        "y": torch.ones(n, length, dtype=torch.float32),
    }
    torch.save(data, Path(directory) / "mock_train_8.pt")
    torch.save(data, Path(directory) / "mock_test_8.pt")


def _load_ptdataset(directory, **kwargs):
    return PTDataset(
        root_dir=directory,
        dataset_name="mock",
        n_train=5,
        n_tests=[5],
        batch_size=1,
        test_batch_sizes=[1],
        train_resolution=8,
        test_resolutions=[8],
        encode_input=False,
        encode_output=False,
        **kwargs,
    )


def test_dtype_none_preserves_float32():
    """dtype=None (default) must not change an existing float32 tensor."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_mock_pt_files(tmpdir, x_dtype=torch.float32)
        ds = _load_ptdataset(tmpdir)
        assert ds.train_db.x.dtype == torch.float32
        assert ds.test_dbs[8].x.dtype == torch.float32


def test_dtype_none_preserves_complex64():
    """dtype=None (default) must preserve complex64 — this was the reported bug."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_mock_pt_files(tmpdir, x_dtype=torch.complex64)
        ds = _load_ptdataset(tmpdir)
        assert ds.train_db.x.dtype == torch.complex64
        assert ds.test_dbs[8].x.dtype == torch.complex64


def test_dtype_explicit_float32_casts_complex64():
    """dtype=torch.float32 must cast complex64 data for callers that need the old behaviour."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_mock_pt_files(tmpdir, x_dtype=torch.complex64)
        ds = _load_ptdataset(tmpdir, dtype=torch.float32)
        assert ds.train_db.x.dtype == torch.float32
        assert ds.test_dbs[8].x.dtype == torch.float32
