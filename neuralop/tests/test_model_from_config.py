
import torch
import time
from tensorly import tenalg
tenalg.set_backend('einsum')
from pathlib import Path
import sys

from neuralop import get_model

# Read the configuration
from zencfg import cfg_from_flat_dict
import sys 
from .test_config import TestConfig


def test_from_config():
    """Test forward/backward from a config file"""
    # Read the configuration
    config = cfg_from_flat_dict(TestConfig, {})
    config = config.to_dict()
    from pprint import pprint

    pprint(config)

    batch_size = config.data.batch_size
    size = config.data.train_resolution

    if torch.has_cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    model = get_model(config)
    model = model.to(device)

    in_data = torch.randn(batch_size, 1, size, size).to(device)
    print(model.__class__)
    print(model)

    t1 = time.time()
    out = model(in_data)
    t = time.time() - t1
    print(f'Output of size {out.shape} in {t}.')

    loss = out.sum()
    loss.backward()
