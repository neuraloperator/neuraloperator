
import torch
import time
from tensorly import tenalg
tenalg.set_backend('einsum')

from neuralop import get_model


# Read the configuration
from zencfg import cfg_from_commandline
import sys 
sys.path.insert(0, '../')
from config.test_config import TestConfig


config = cfg_from_commandline(TestConfig)
print(config)
print(config.to_dict())
config = config.to_dict()

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

# Check for unused params
for name, param in model.named_parameters():
    if param.grad is None:
        print(f'Usused parameter {name}!')
