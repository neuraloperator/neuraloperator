import time
from ..uno import UNO
import torch
import pytest


@pytest.mark.parametrize(
    "input_shape", [(32, 3, 64, 55), (32, 3, 100, 105), (32, 3, 133, 95)]
)
def test_UNO(input_shape):
    horizontal_skips_map ={4:0,3:1}
    model = UNO(3,3,5,uno_out_channels = [32,64,64,64,32], 
                uno_n_modes= [[5,5],[5,5],[5,5],[5,5],[5,5]], 
                uno_scalings=  [[1.0,1.0],[0.5,0.25],[1,1],[1,1],[2,4]],\
                 horizontal_skips_map = horizontal_skips_map, 
                 n_layers = 5, domain_padding = 0.2, 
                 channel_mlp_skip="linear",
                 resolution_scaling_factor = 1)

    t1 = time.time()
    in_data = torch.randn(input_shape)
    out = model(in_data)
    t = time.time() - t1
    print(f"Output of size {out.shape} in {t}.")
    for i in range(len(out.shape)):
        assert in_data.shape[i] == out.shape[i]
    loss = out.sum()
    t1 = time.time()
    loss.backward()
    t = time.time() - t1
    print(f"Gradient Calculated in {t}.")
    n_unused_params = 0

    for name, param in model.named_parameters():
        if param.grad is None:
            n_unused_params += 1

    model = UNO(
        3,
        3,
        5,
        uno_out_channels=[32, 64, 64, 64, 32],
        uno_n_modes=[[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]],
        uno_scalings=[[1.0, 1.0], [0.5, 0.5], [1, 1], [1, 1], [2, 2]],
        horizontal_skips_map=None,
        n_layers=5,
        domain_padding=0.2,
        resolution_scaling_factor=1,
        channel_mlp_skip="linear" 
    )

    t1 = time.time()
    in_data = torch.randn(input_shape)
    out = model(in_data)
    t = time.time() - t1
    print(f"Output of size {out.shape} in {t}.")

    loss = out.sum()
    t1 = time.time()
    loss.backward()
    t = time.time() - t1
    print(f"Gradient Calculated in {t}.")
    n_unused_params = 0

    for name, param in model.named_parameters():
        if param.grad is None:
            n_unused_params += 1

    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"
