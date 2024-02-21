import pytest
import torch
from ..attention_kernel_integral import AttentionKernelIntegral
from ..embeddings import RotaryEmbedding


def test_AttentionWithRoPE():
    """Test for Attention kernel integral with rotary position embeddings

    Checks the output size
    """
    head_n_channels = 32
    channels = 32
    num_heads = 4
    n_dim = 2
    num_points = 64
    batch_size = 4

    attn_layer = AttentionKernelIntegral(channels, channels, num_heads, head_n_channels)
    pos_emb_module = RotaryEmbedding(head_n_channels//n_dim)

    x = torch.randn(batch_size, num_points, channels)
    pos = torch.randn(batch_size, num_points, n_dim)

    freqs = pos_emb_module(pos)
    assert freqs.shape == (batch_size, num_points, head_n_channels)

    res = attn_layer(x, pos, positional_embedding_module=pos_emb_module)
    assert res.shape == (batch_size, num_points, channels)