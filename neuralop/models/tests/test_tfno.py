
import torch
from neuralop import TFNO3d, TFNO2d, TFNO1d, TFNO
import pytest
from tensorly import tenalg
from configmypy import Bunch
tenalg.set_backend('einsum')


@pytest.mark.parametrize('factorization', ['ComplexDense', 'ComplexTucker', 'ComplexCP', 'ComplexTT'])
@pytest.mark.parametrize('implementation', ['factorized', 'reconstructed'])
@pytest.mark.parametrize('n_dim', [1, 2, 3])
def test_tfno(factorization, implementation, n_dim):
    if torch.has_cuda:
        device = 'cuda'
        s = 128
        modes = 16
        width = 64
        fc_channels = 256
        batch_size = 20
        use_mlp = True
        n_layers = 4
        mlp = Bunch(dict(expansion=0.5, dropout=0))
    else:
        device = 'cpu'
        s = 16
        modes = 5
        width = 15
        fc_channels = 32
        batch_size = 3
        n_layers = 2

        use_mlp = True
        mlp = Bunch(dict(expansion=0.5, dropout=0))

    rank = 0.2
    size = (s, )*n_dim
    m_modes = (modes,)*n_dim
    model = TFNO(hidden_channels=width, n_modes=m_modes, 
                            factorization=factorization,
                            implementation=implementation,
                            rank=rank,
                            fixed_rank_modes=False,
                            joint_factorization=False,
                            n_layers=n_layers,
                            use_mlp=use_mlp, mlp=mlp,
                            fc_channels=fc_channels).to(device)
    in_data = torch.randn(batch_size, 3, *size).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]

    # Check backward pass
    loss = out.sum()
    loss.backward()
