import pytest
import torch
from ..codano import TnoBlock2d
from ..spectral_convolution import *


@pytest.mark.parametrize('token_codimension', 
                         [1,2,5])
def test_Codano(token_codimension):
    """
    Test CoDA-NO layers 
    """
    n_modes = [100, 100]
    n_head=3
    token_codimension=2
    output_scaling_factor=None
    incremental_n_modes=None
    head_codimension=1
    SpectralConvolution=SpectralConv
    Normalizer=None
    joint_factorization=False
    fixed_rank_modes=False
    implementation='factorized'
    decomposition_kwargs=None
    fft_norm='forward'
    codimension_size=None
    per_channel_attention=True
    permutation_eq=True
    temperature=1.0
    kqv_non_linear=False

    layer = TnoBlock2d(
        n_modes=n_modes,
        n_head=n_head,
        token_codimension=token_codimension,
        output_scaling_factor=output_scaling_factor,
        incremental_n_modes=incremental_n_modes,
        head_codimension=head_codimension,
        SpectralConvolution=SpectralConvolution,
        Normalizer=Normalizer,
        joint_factorization=joint_factorization,
        fixed_rank_modes=fixed_rank_modes,
        implementation=implementation,
        decomposition_kwargs=decomposition_kwargs,
        fft_norm=fft_norm,
        codimension_size=codimension_size,
        per_channel_attention=per_channel_attention,
        permutation_eq=permutation_eq,
        temperature=temperature,
        kqv_non_linear=kqv_non_linear,
    )

    x = torch.randn(2, 10, 128, 128)

    out = layer(x)

    assert out.shape == x.shape
    
    