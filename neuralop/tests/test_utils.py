from ..utils import get_wandb_api_key, wandb_login
from ..utils import count_model_params, count_tensor_params
from pathlib import Path
import pytest
import wandb
import os
import torch
from torch import nn

def test_count_model_params():
    # A nested dummy model to make sure all parameters are counted 
    class DumyModel(nn.Module):
        def __init__(self, n_submodels=0, dtype=torch.float32):
            super().__init__()

            self.n_submodels = n_submodels
            self.param = nn.Parameter(torch.randn((2, 3, 4), dtype=dtype))
            if n_submodels:
                self.model = DumyModel(n_submodels - 1, dtype=dtype)
    
    n_submodels = 2
    model = DumyModel(n_submodels=n_submodels)
    n_params = count_model_params(model)
    print(model)
    assert n_params ==  (n_submodels+1) * 2*3*4

    model = DumyModel(n_submodels=n_submodels, dtype=torch.cfloat)
    n_params = count_model_params(model)
    print(model)
    assert n_params ==  2 * (n_submodels+1) * 2*3*4


def test_count_tensor_params():
    # Case 1 : real tensor
    x = torch.randn((2, 3, 4, 5, 6), dtype=torch.float32)

    # dims = None: count all params
    n_params = count_tensor_params(x)
    assert n_params ==  2*3*4*5*6
    # Only certain dims
    n_params = count_tensor_params(x, dims=[1, 3])
    assert n_params ==  3*5

    # Case 2 : complex tensor
    x = torch.randn((2, 3, 4, 5, 6), dtype=torch.cfloat)

    # dims = None: count all params
    n_params = count_tensor_params(x)
    assert n_params ==  2*3*4*5*6 * 2
    # Only certain dims
    n_params = count_tensor_params(x, dims=[1, 3])
    assert n_params ==  3*5 * 2



def test_get_wandb_api_key():
    # Make sure no env var key set
    os.environ.pop("WANDB_API_KEY", None)

    # Read from file
    filepath = Path(__file__).parent.joinpath('test_config_key.txt').as_posix()
    key = get_wandb_api_key(filepath)
    assert key == 'my_secret_key'

    # Read from env var
    os.environ["WANDB_API_KEY"] = 'key_from_env'
    key = get_wandb_api_key(filepath)
    assert key == 'key_from_env'

    # Read from env var with incorrect file
    os.environ["WANDB_API_KEY"] = 'key_from_env'
    key = get_wandb_api_key('wrong_path')
    assert key == 'key_from_env'


def test_ArgparseConfig(monkeypatch):
    def login(key):
        if key == 'my_secret_key':
            return True

        raise ValueError('Wrong key')

    monkeypatch.setattr(wandb, "login", login)

    # Make sure no env var key set
    os.environ.pop("WANDB_API_KEY", None)

    # Read from file
    filepath = Path(__file__).parent.joinpath('test_config_key.txt').as_posix()
    assert wandb_login(filepath) is None

    # Read from env var
    os.environ["WANDB_API_KEY"] = 'my_secret_key'
    assert wandb_login() is None

    # Read from env var
    os.environ["WANDB_API_KEY"] = 'wrong_key'
    assert wandb_login(key='my_secret_key') is None

    # Read from env var
    os.environ["WANDB_API_KEY"] = 'wrong_key'
    with pytest.raises(ValueError):
        wandb_login()

