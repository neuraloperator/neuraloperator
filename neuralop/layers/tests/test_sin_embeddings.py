import random

import torch
from torch.testing import assert_close
import pytest

from ..embeddings import NeRFEmbedding

# Testing NeRF Embedding: start with a simple range 
# and see that it is embedded properly

def test_NeRFEmbedding():
    pos_embed = NeRFEmbedding(num_frequencies=3)
    unbatched_inputs = torch.ones(3) * torch.tensor([[1.], [0.5]])
    embeds = pos_embed(unbatched_inputs)
    print(embeds)

    assert embeds.shape == (2,pos_embed.out_channels(4))
