import torch
from torch.autograd import grad
import pytest
from tensorly import tenalg

tenalg.set_backend("einsum")

from ..fnogno import FNOGNO


@pytest.mark.parametrize(
    "gno_transform_type", ["linear", "nonlinear_kernelonly", "nonlinear"]
)
@pytest.mark.parametrize("fno_n_modes", [(8,), (8, 8), (8, 8, 8)])
@pytest.mark.parametrize("gno_batched", [False, True])
def test_fnogno(gno_transform_type, fno_n_modes, gno_batched):
    if torch.has_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu:0")

    in_channels = 3
    out_channels = 2
    batch_size = 4
    n_dim = len(fno_n_modes)
    model = FNOGNO(
        in_channels=in_channels,
        out_channels=out_channels,
        gno_radius=0.2,
        gno_coord_dim=n_dim,
        gno_transform_type=gno_transform_type,
        gno_batched=gno_batched,
        fno_n_modes=fno_n_modes,
        fno_norm="ada_in",
        fno_ada_in_features=4,
    ).to(device)

    in_p_shape = [
        32,
    ] * n_dim
    in_p_shape.append(n_dim)
    in_p = torch.randn(*in_p_shape).to(device)

    out_p = torch.randn(100, n_dim).to(device)

    f_shape = [
        32,
    ] * n_dim
    f_shape.append(in_channels)

    # repeat along batch dim if batched
    if gno_batched:
        f_shape = [batch_size] + f_shape

    f = torch.randn(*f_shape).to(device).requires_grad_(True)
    ada_in = torch.randn(
        1,
    ).to(device)

    # Test forward pass
    out = model(in_p, out_p, f, ada_in)

    # Check output size
    if gno_batched:
        assert list(out.shape) == [batch_size, 100, out_channels]

        # take grad of out wrt feature inputs
        out0_f_grad = grad(outputs=out[0].sum(), inputs=f, retain_graph=True)[0]

        # check the first output example has grad wrt first f example
        assert out0_f_grad[0].nonzero().any()

        # check the first output example has no grad wrt second f example
        assert not out0_f_grad[1].nonzero().any()
        del out0_f_grad

    else:
        assert list(out.shape) == [100, out_channels]

    # Check backward pass
    loss = out.sum()
    loss.backward()

    n_unused_params = 0
    for param in model.parameters():
        if param.grad is None:
            n_unused_params += 1
    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"
