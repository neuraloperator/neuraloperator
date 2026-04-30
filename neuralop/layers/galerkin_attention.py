"""Galerkin-style linear attention via spectral kernel methods.

This module provides GalerkinAttention, a linearized attention mechanism
that replaces O(n^2 d) softmax attention with an O(n d^2) spectral kernel
computed in the frequency domain. Inspired by Galerkin methods for PDEs.

Compatible with the neuraloperator layers API (in_channels, out_channels,
n_heads, head_n_channels) and supports optional rotary positional embeddings.
"""
import torch
from torch import nn
import math


class GalerkinAttention(nn.Module):
    """Galerkin-style linear attention via spectral kernel methods.

    Replaces the O(n^2 d) softmax attention with an O(n d^2) linearized
    attention kernel computed in the frequency domain, inspired by the
    Galerkin method for solving PDEs. The kernel is obtained by projecting
    query-key interactions onto a low-rank frequency basis.

    Specifically computes:
        K(x,y) ~ sum_{m=1}^{M} qhat_m(x) * khat_m(y)*
    where M = n_modes limits the number of frequency modes, and the
    resulting kernel is applied to values v(y) via irfft.

    This is complementary to AttentionKernelIntegral which uses quadrature-
    based kernel integration. Here the kernel is implicit (spectral) rather
    than assembled explicitly.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    n_heads : int
        Number of attention heads.
    head_n_channels : int
        Dimension of each attention head (d // n_heads).
    n_modes : int, optional
        Number of low-frequency Fourier modes to retain. Default is 16.
    project_query : bool, optional
        Whether to project query with a linear layer. Default True.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_heads,
        head_n_channels,
        n_modes=16,
        project_query=True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_n_channels = head_n_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.project_query = project_query

        if project_query:
            self.to_q = nn.Linear(in_channels, head_n_channels * n_heads, bias=False)
        else:
            self.to_q = nn.Identity()

        self.to_k = nn.Linear(in_channels, head_n_channels * n_heads, bias=False)
        self.to_v = nn.Linear(in_channels, head_n_channels * n_heads, bias=False)

        self.to_out = (
            nn.Linear(head_n_channels * n_heads, out_channels)
            if head_n_channels * n_heads != out_channels
            else nn.Identity()
        )

        self.alpha = nn.Parameter(torch.ones(1))

    def forward(
        self,
        u_src,
        pos_src=None,
        positional_embedding_module=None,
        u_qry=None,
        pos_qry=None,
        weights=None,
        return_kernel=False,
    ):
        """Forward pass for Galerkin-style linear attention.

        Parameters
        ----------
        u_src : torch.Tensor
            Source input [batch_size, num_points, in_channels].
        pos_src : torch.Tensor, optional
            Source positions [batch_size, num_points, pos_dim].
        positional_embedding_module : nn.Module, optional
            Rotary embedding module (e.g. RotaryEmbedding2D).
        u_qry : torch.Tensor, optional
            Query input. If None, defaults to u_src (self-attention).
        return_kernel : bool, optional
            If True, return (output, kernel).

        Returns
        -------
        torch.Tensor or tuple
        """
        if u_qry is None:
            u_qry = u_src

        batch_size, num_points, _ = u_src.shape
        pos_dim = pos_src.shape[-1] if pos_src is not None else 0

        q = self.to_q(u_qry)
        k = self.to_k(u_src)
        v = self.to_v(u_src)

        q = q.view(batch_size, -1, self.n_heads, self.head_n_channels)
        k = k.view(batch_size, -1, self.n_heads, self.head_n_channels)
        v = v.view(batch_size, -1, self.n_heads, self.head_n_channels)

        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        if positional_embedding_module is not None:
            if pos_dim == 2:
                pos_x = pos_src[..., 0]
                pos_y = pos_src[..., 1]
                k_freqs_x = positional_embedding_module(pos_x)
                k_freqs_y = positional_embedding_module(pos_y)

                if u_qry is u_src and pos_qry is None:
                    q_freqs_x, q_freqs_y = k_freqs_x, k_freqs_y
                else:
                    pq_x = pos_qry[..., 0] if pos_qry is not None else pos_x
                    pq_y = pos_qry[..., 1] if pos_qry is not None else pos_y
                    q_freqs_x = positional_embedding_module(pq_x)
                    q_freqs_y = positional_embedding_module(pq_y)

                q = _apply_rotary_2d(q, q_freqs_x, q_freqs_y)
                k = _apply_rotary_2d(k, k_freqs_x, k_freqs_y)
            elif pos_dim == 1:
                k_freqs = positional_embedding_module(pos_src[..., 0])
                q_freqs = positional_embedding_module(
                    pos_qry[..., 0] if pos_qry is not None else pos_src[..., 0]
                )
                q = _apply_rotary_1d(q, q_freqs)
                k = _apply_rotary_1d(k, k_freqs)

        q_fft = torch.fft.rfft(q, dim=2, norm="ortho")
        k_fft = torch.fft.rfft(k, dim=2, norm="ortho")

        n_freqs = q_fft.shape[2]
        M = min(self.n_modes, n_freqs)

        q_m = q_fft[:, :, :M, :]
        k_m = k_fft[:, :, :M, :]

        kernel = torch.einsum("bhnm, bhnm -> bhnm", q_m, k_m.conj())
        kernel = kernel * self.alpha

        kernel_conj_flip = kernel.conj().flip(-2)
        kernel_full = torch.cat([kernel, kernel_conj_flip], dim=2)[:, :, :n_freqs, :]

        attn = torch.fft.irfft(kernel_full, n=num_points, dim=2, norm="ortho")

        if weights is not None:
            attn = attn * weights.view(1, 1, -1, 1)

        out = attn * v
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, num_points, self.n_heads * self.head_n_channels)
        out = self.to_out(out)

        if return_kernel:
            return out, kernel
        return out


def _apply_rotary_1d(x, freqs):
    """Apply 1D rotary positional embedding."""
    d = x.shape[-1] // 2
    x_real, x_imag = x[..., :d], x[..., d:]
    x_rot_real = x_real * freqs - x_imag * freqs.flipud()
    x_rot_imag = x_real * freqs.flipud() + x_imag * freqs
    return torch.cat([x_rot_real, x_rot_imag], dim=-1)


def _apply_rotary_2d(x, freqs_x, freqs_y):
    """Apply 2D rotary positional embedding (RoPE)."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    d1, d2 = d // 2, d // 2
    x1r, x1i = x1[..., :d1], x1[..., d1:]
    x2r, x2i = x2[..., :d2], x2[..., d2:]
    fx1 = freqs_x.unsqueeze(-1)
    fx2 = freqs_y.unsqueeze(-1)
    x1r_out = x1r * fx1 - x1i * fx1.flipud()
    x1i_out = x1r * fx1.flipud() + x1i * fx1
    x2r_out = x2r * fx2 - x2i * fx2.flipud()
    x2i_out = x2r * fx2.flipud() + x2i * fx2
    return torch.cat([torch.cat([x1r_out, x1i_out], -1), torch.cat([x2r_out, x2i_out], -1)], -1)
