import pytest
import torch
import time
from tensorly import tenalg

tenalg.set_backend("einsum")
from pathlib import Path
import sys

from neuralop import get_model

# Read the configuration
from zencfg.from_commandline import make_config
import sys
from .test_config import TestConfig


def test_from_config():
    """Test forward/backward from a config file"""
    # Read the configuration
    config = make_config(TestConfig)
    config = config.to_dict()
    from pprint import pprint

    pprint(config)

    batch_size = config.data.batch_size
    size = config.data.train_resolution

    if torch.has_cuda:
        device = "cuda"
    else:
        device = "cpu"

    model = get_model(config)
    model = model.to(device)

    in_data = torch.randn(batch_size, 1, size, size).to(device)
    print(model.__class__)
    print(model)

    t1 = time.time()
    out = model(in_data)
    t = time.time() - t1
    print(f"Output of size {out.shape} in {t}.")

    loss = out.sum()
    loss.backward()


@pytest.mark.parametrize("include_config_name", [False, True])
def test_model_config_metadata(include_config_name):
    config = make_config(TestConfig).to_dict()
    if include_config_name:
        config.model["_config_name"] = "FNO_Small2d"
    else:
        config.model.pop("_config_name", None)
    original_model_config = config.model.copy()

    model = get_model(config)

    assert config.model == original_model_config
    assert "_config_name" not in model._init_kwargs
    assert model.fno_blocks.norm is None
    assert model.factorization is None
    assert model.fno_blocks.stabilizer is None
