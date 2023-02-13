
import torch
import time
from tensorly import tenalg
tenalg.set_backend('einsum')

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from neuralop import get_model


# Read the configuration
config_name = 'default'
pipe = ConfigPipeline([YamlConfig('./test_config.yaml', config_name='default', config_folder='../config'),
                       ArgparseConfig(infer_types=True, config_name=None, config_file=None),
                       YamlConfig(config_folder='../config')
                      ])
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

batch_size = config.data.batch_size
size = config.data.size

if torch.has_cuda:
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
