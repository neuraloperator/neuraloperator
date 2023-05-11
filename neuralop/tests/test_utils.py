from ..utils import get_wandb_api_key
from pathlib import Path

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
