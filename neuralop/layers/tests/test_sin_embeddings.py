import math

import torch
from torch.testing import assert_close
import pytest

from ..embeddings import SinusoidalEmbedding

# Testing NeRF Embedding: start with a simple range 
# and see that it is embedded properly

batch_size = 4
num_freqs = 3
in_channels = 3
n_in = 2
max_pos = 10000

if torch.backends.cuda.is_built():
    device = "cuda"
else:
    device = "cpu"

def test_NeRFEmbedding():
    nerf_embed = SinusoidalEmbedding(in_channels=in_channels,
                                    num_frequencies=3,
                                    embedding_type='nerf')
    unbatched_inputs = torch.arange(in_channels) * torch.tensor([[1.], [0.5]])
    embeds = nerf_embed(unbatched_inputs.to(device))

    true_outputs = torch.zeros(n_in, in_channels * num_freqs * 2).to(device)

    # True values are (sin(2^0 * pi * p), cos(2^0 * pi * p), ... cos(2^(L-1) * pi * p))
    for channel in range(in_channels):
        for wavenumber in range(num_freqs):
            for i in range(2):
                idx = channel * (num_freqs * 2) + wavenumber * 2 + i
                freqs = 2 ** wavenumber * math.pi * unbatched_inputs[:, channel]
                if i == 0:
                    true_outputs[:, idx] = freqs.sin()
                else:
                    true_outputs[:, idx] = freqs.cos()
    assert_close(embeds, true_outputs)

    batched_inputs = torch.stack([torch.arange(in_channels) * torch.tensor([[1.], [0.5]])] * batch_size)
    embeds = nerf_embed(batched_inputs.to(device))

    true_outputs = torch.zeros(batch_size, n_in, in_channels * num_freqs * 2).to(device)

    # True values are (sin(2^0 * pi * p), cos(2^0 * pi * p), ... cos(2^(L-1) * pi * p))
    for channel in range(in_channels):
        for wavenumber in range(num_freqs):
            for i in range(2):
                idx = channel * (num_freqs * 2) + wavenumber * 2 + i
                freqs = 2 ** wavenumber * math.pi * batched_inputs[:, :, channel]
                if i == 0:
                    true_outputs[:, :, idx] = freqs.sin()
                else:
                    true_outputs[:, :, idx] = freqs.cos()
    assert_close(embeds, true_outputs)


def test_TransformerEmbedding():
    nerf_embed = SinusoidalEmbedding(in_channels=in_channels,
                                    num_frequencies=3,
                                    embedding_type='transformer',
                                    max_positions=max_pos)
    unbatched_inputs = torch.arange(in_channels) * torch.tensor([[1.], [0.5]])
    embeds = nerf_embed(unbatched_inputs.to(device))

    true_outputs = torch.zeros(n_in, in_channels * num_freqs * 2).to(device)

    # True values are (sin(2^0 * pi * p), cos(2^0 * pi * p), ... cos(2^(L-1) * pi * p))
    for channel in range(in_channels):
        for wavenumber in range(num_freqs):
            for i in range(2):
                idx = channel * (num_freqs * 2) + wavenumber * 2 + i
                freqs = ((1 / max_pos) ** (wavenumber / in_channels)) * unbatched_inputs[:, channel]
                if i == 0:
                    true_outputs[:, idx] = freqs.sin()
                else:
                    true_outputs[:, idx] = freqs.cos()
    assert_close(embeds, true_outputs)

    batched_inputs = torch.stack([torch.arange(in_channels) * torch.tensor([[1.], [0.5]])] * batch_size)
    embeds = nerf_embed(batched_inputs.to(device))

    true_outputs = torch.zeros(batch_size, n_in, in_channels * num_freqs * 2).to(device)

    # True values are (sin(2^0 * pi * p), cos(2^0 * pi * p), ... cos(2^(L-1) * pi * p))
    for channel in range(in_channels):
        for wavenumber in range(num_freqs):
            for i in range(2):
                idx = channel * (num_freqs * 2) + wavenumber * 2 + i
                freqs = ((1 / max_pos) ** (wavenumber / in_channels)) * batched_inputs[:, :, channel]
                if i == 0:
                    true_outputs[:, :, idx] = freqs.sin()
                else:
                    true_outputs[:, :, idx] = freqs.cos()
    assert_close(embeds, true_outputs)