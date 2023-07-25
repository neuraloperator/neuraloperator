from pathlib import Path
import time

from configmypy import ConfigPipeline, YamlConfig
import torch

from tensorly import tenalg
from neuralop import get_model
tenalg.set_backend('einsum')


def test_from_config():
    """Test forward/backward from a config file"""
    # Read the configuration
    config_name = 'default'
    config_path = Path(__file__).parent.as_posix()
    pipe = ConfigPipeline(
        [YamlConfig(
            './test_config.yaml',
            config_name=config_name,
            config_folder=config_path)])
    config = pipe.read_conf()
    config_name = pipe.steps[-1].config_name

    batch_size = config.data.batch_size
    size = config.data.size

    if torch.backends.cuda.is_built():
        device = 'cuda'
    else:
        device = 'cpu'

    model = get_model(config)
    model = model.to(device)

    in_data = torch.randn(batch_size, 3, size, size).to(device)
    print(model.__class__)
    print(model)

    t1 = time.time()
    out = model(in_data)
    t = time.time() - t1
    print(f'Output of size {out.shape} in {t}.')

    loss = out.sum()
    loss.backward()
