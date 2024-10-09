import pytest
import torch
from ..coda_blocks import *
from ..spectral_convolution import *


@pytest.mark.parametrize('token_codimension',
                         [1, 2, 5])
def test_Codano(token_codimension):
    """
    Test CoDA-NO layers
    """
    n_modes_2D = [64, 64]
    n_modes_3D = [64, 64, 64]
    n_head = 3
    output_scaling_factor = None
    incremental_n_modes = None
    head_codimension = 3
    spectral_convolution = SpectralConv
    Normalizer = None
    joint_factorization = False
    fixed_rank_modes = False
    implementation = 'factorized'
    decomposition_kwargs = None
    fft_norm = 'forward'
    codimension_size = None
    per_channel_attention = False
    permutation_eq = True
    temperature = 1.0
    kqv_non_linear = False

    layer = CODABlocks(
        n_modes=n_modes_2D,
        n_head=n_head,
        token_codimension=token_codimension,
        output_scaling_factor=output_scaling_factor,
        incremental_n_modes=incremental_n_modes,
        head_codimension=head_codimension,
        spectral_convolution=spectral_convolution,
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

    # test different spatial resolution 
    x = torch.randn(2, 10, 100, 100)
    out = layer(x)
    assert out.shape == x.shape

    

    layer = CODABlocks(
        n_modes=n_modes_3D,
        n_head=n_head,
        token_codimension=token_codimension,
        output_scaling_factor=output_scaling_factor,
        incremental_n_modes=incremental_n_modes,
        head_codimension=head_codimension,
        spectral_convolution=spectral_convolution,
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

    x = torch.randn(2, 10, 128, 128, 128)
    out = layer(x)
    assert out.shape == x.shape

    # test different spato-temporal resolution
    x = torch.randn(2, 10, 100, 100, 100)
    out = layer(x)
    assert out.shape == x.shape
