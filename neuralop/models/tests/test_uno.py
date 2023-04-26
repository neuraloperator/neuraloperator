import time
from ..uno import UNO
import torch

def test_UNO():
    horizontal_skips_map ={4:0,3:1}
    model = UNO(3,3,5,uno_out_channels = [32,64,64,64,32], uno_n_modes= [[5,5],[5,5],[5,5],[5,5],[5,5]], uno_scalings=  [[1.0,1.0],[0.5,0.5],[1,1],[1,1],[2,2]],\
                 horizontal_skips_map = horizontal_skips_map, n_layers = 5, domain_padding = 0.2)

    t1 = time.time()
    in_data = torch.randn(32,3,64,64)
    out = model(in_data)
    out = model(in_data)
    out = model(in_data)
    t = time.time() - t1
    print(f'Output of size {out.shape} in {t}.')

    loss = out.sum()
    t1 = time.time()
    loss.backward()
    t = time.time() - t1
    print(f'Gradient Calculated in {t}.')
    n_unused_params = 0

    for name,param in model.named_parameters():
        if param.grad is None:
            n_unused_params += 1


    model = UNO(3,3,5,uno_out_channels = [32,64,64,64,32], uno_n_modes= [[5,5],[5,5],[5,5],[5,5],[5,5]], uno_scalings=  [[1.0,1.0],[0.5,0.5],[1,1],[1,1],[2,2]],\
                 horizontal_skips_map = None, n_layers = 5, domain_padding = 0.2)

    t1 = time.time()
    in_data = torch.randn(32,3,64,64)
    out = model(in_data)
    out = model(in_data)
    out = model(in_data)
    t = time.time() - t1
    print(f'Output of size {out.shape} in {t}.')

    loss = out.sum()
    t1 = time.time()
    loss.backward()
    t = time.time() - t1
    print(f'Gradient Calculated in {t}.')
    n_unused_params = 0

    for name,param in model.named_parameters():
        if param.grad is None:
            n_unused_params += 1

    assert n_unused_params == 0, f'{n_unused_params} parameters were unused!'