import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
def rotate_half(x):
    """
    Split x's channels into two equal halves.
    """
    # split the last dimension of x into two equal halves
    x = x.reshape(*x.shape[:-1], 2, -1)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    Apply rotation matrix computed based on freqs to rotate t.
    t: tensor of shape [batch_size, num_points, dim]
    freqs: tensor of shape [batch_size, num_points, 1]

    Formula: see equation (34) in https://arxiv.org/pdf/2104.09864.pdf
    """
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1/64, scale=1.):
        """
        Applying rotary positional embedding (https://arxiv.org/abs/2104.09864) to the input feature tensor.
        The crux is the dot product of two rotation matrices R(theta1) and R(theta2) is equal to R(theta2 - theta1).
        """
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, coordinates):
        """coordinates is tensor of [batch_size, num_points]"""
        coordinates = coordinates * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', coordinates, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]

    @staticmethod
    def apply_1d_rotary_pos_emb(t, freqs):
        return apply_rotary_pos_emb(t, freqs)

    @staticmethod
    def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
        """Split the last dimension of features into two equal halves
           and apply 1d rotary positional embedding to each half."""
        d = t.shape[-1]
        t_x, t_y = t[..., :d//2], t[..., d//2:]

        return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                          apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)