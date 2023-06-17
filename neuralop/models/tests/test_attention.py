import time
from ..attention import TnoBlock2d
import torch
import pytest

@pytest.mark.parametrize('input_shape', 
                         [(32,4,64,55),(32,4,100,105),(32,4,133,95)]) #
def test_attention(input_shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TnoBlock2d(4, 1, [20,20], token_codim = 2).to(device)
    t1 = time.time()
    in_data = torch.randn(input_shape).to(device)
    out = model(in_data)
    t = time.time() - t1
    print(f'Output of size {out.shape} in {t}.')
    for i in range(len(out.shape)):
        assert in_data.shape[i] == out.shape[i]
    loss = out.sum()
    t1 = time.time()
    with torch.autograd.set_detect_anomaly(True):
        loss.backward()
    t = time.time() - t1
    print(f'Gradient Calculated in {t}.')
    n_unused_params = 0

    for name,param in model.named_parameters():
        if param.grad is None:
            n_unused_params += 1

    assert n_unused_params == 0, f'{n_unused_params} parameters were unused!'
    
    model = TnoBlock2d(4, 3, [20,20], token_codim = 2).to(device)
    t1 = time.time()
    in_data = torch.randn(input_shape).to(device)
    out = model(in_data)
    t = time.time() - t1
    print(f'Output of size {out.shape} in {t}.')
    for i in range(len(out.shape)):
        assert in_data.shape[i] == out.shape[i]
    loss = out.sum()
    t1 = time.time()
    with torch.autograd.set_detect_anomaly(True):
        loss.backward()
    t = time.time() - t1
    print(f'Gradient Calculated in {t}.')
    n_unused_params = 0

    for name,param in model.named_parameters():
        if param.grad is None:
            n_unused_params += 1

    assert n_unused_params == 0, f'{n_unused_params} parameters were unused!'