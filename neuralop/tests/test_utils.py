from ..utils import get_wandb_api_key, wandb_login
from pathlib import Path
import pytest
import wandb
import os

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