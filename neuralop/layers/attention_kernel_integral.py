import torch
from torch import nn
import math
from torch.nn.init import xavier_uniform_, zeros_


class AttentionKernelIntegral(torch.nn.Module):
    """
    Kernel integral transform with attention
    Computes \int_{Omega} k(x, y) * f(y) dy,
    where:
          K(x, y) = \sum_{c=1}^d \q_c(x) * \k_c(y)
          f(y) = v(y)

    Parameters
    ----------
    in_channels : int, input channels
    out_channels : int, output channels
    n_heads : int, number of attention heads in multi-head attention
    head_n_channels : int, dimension of each attention head, determines how many function bases to use for the kernel
                      k(x, y) = \sum_{c=1}^d \q_c(x) * \k_c(y), head_n_channels controls the d
    pos_dim : int, dimension of the domain, determines the dimension of coordinates
    use_positional_encoding : bool, whether to use positional encoding to encode query and key
    project_query : bool, whether to project the query function with pointwise linear layer
                   (this is sometimes not needed when using cross-attention)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_heads,
                 head_n_channels,
                 pos_dim,
                 use_positional_encoding=True,    # use positional encoding
                 project_query=True,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.n_heads = n_heads
        self.head_n_channels = head_n_channels

        self.project_query = project_query
        if project_query:
            self.to_q = nn.Linear(in_channels, head_n_channels * n_heads, bias=False)
        else:
            self.to_q = nn.Identity()

        self.to_k = nn.Linear(in_channels, head_n_channels * n_heads, bias=False)

        self.k_norm = nn.InstanceNorm1d(head_n_channels, affine=False)
        self.v_norm = nn.InstanceNorm1d(head_n_channels, affine=False)

        self.to_v = nn.Linear(in_channels, head_n_channels * n_heads, bias=False)

        self.to_out = nn.Linear(head_n_channels * n_heads, out_channels) \
            if head_n_channels * n_heads != out_channels else nn.Identity()

        self.use_positional_encoding = use_positional_encoding
        self.pos_dim = pos_dim

        self.init_gain = 1 / math.sqrt(head_n_channels)
        self.diagonal_weight = self.init_gain
        self.initialize_qkv_weights()

    def init_weight(self, weight, inif_fn):
        """
        Initialization for the projection matrix
        basically initialize the weights for each heads with predefined initialization function and gain,
        to add the diagonal bias, it requires input channels = head_n_channels
        W = init_fn(W) + I * diagonal_weight

        init_fn is xavier_uniform_ by default
        """

        for param in weight.parameters():
            if param.ndim > 1:
                for h in range(self.n_heads):
                    inif_fn(param[h * self.head_n_channels:(h + 1) * self.head_n_channels, :], gain=self.init_gain)
                    if self.head_n_channels == self.channels:
                        diagonal_bias = self.diagonal_weight * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))
                        param.data[h * self.head_n_channels:(h + 1) * self.head_n_channels, :] += diagonal_bias

    def initialize_qkv_weights(self):
        """
        Initialize the weights for q, k, v projection matrix with a small gain and add a diagonal bias,
        this technique has been found useful for scale-sensitive problem that has not been normalized
        see Table 8 in https://arxiv.org/pdf/2105.14995.pdf
        """
        init_fn = xavier_uniform_

        if self.project_query:
            self.init_weight(self.to_q, init_fn)
        self.init_weight(self.to_k, init_fn)
        self.init_weight(self.to_v, init_fn)

    def normalize_wrt_domain(self, u, norm_fn):
        """
        Normalize the input function with respect to the domain
        The second dimension is equal to the number of grid points that discretize the domain
        """
        # u: the input or transformed function
        batch_size = u.shape[0]
        u = u.view(batch_size*self.n_heads, -1, self.head_n_channels)
        u = norm_fn(u)    # layer norm with channel dimension or instance norm with spatial dimension
        return u.view(batch_size, self.n_heads, -1, self.head_n_channels)

    def forward(self,
                u_x,
                pos_x,
                positional_embedding_module=None,    # positional encoding module for encoding q/k
                u_y=None,
                pos_y=None,
                weights=None,
                associative=True,   # can be much faster if num_grid_points is larger than the channel number c
                return_kernel=False):
        """
        Computes kernel integral transform with attention

        Parameters
        ----------
        u_x: input (query) function of shape [batch_size, num_grid_points, channels]
        pos_x: coordinate of input function's grid points [batch_size, num_grid_points, pos_dim]
        positional_embedding_module: positional embedding module for encoding query/key (q/k), a torch.nn.Module
        u_y: the second source of function (key and value), if not provided, u_y = u_x
        pos_y: coordinate of the second source of function's grid points, if not provided, assume pos_y = pos_x
        weights : tensor of shape [batch_size, num_grid_points], if not provided assume to be 1/num_grid_points
                  Weights for each point y proportional to the
                  volume around f(y)=u_y W_v being integrated.
        associative: if True, use associativity of matrix multiplication, first multiply K^T V, then multiply Q,
                   much faster when num_grid_points is larger than the channel number (which is usually the case)
        return_kernel: if True, return the kernel matrix (for analyzing the kernel)

        Output
        ----------
        out_features: Output function given on the points x.
        """

        if u_y is None:
            u_y = u_x   # go back to self attention

        if return_kernel and associative:
            raise Exception('Cannot get kernel matrix when associative is set to True')

        batch_size, num_grid_points = u_y.shape[:2]   # batch size and number of grid points

        q = self.to_q(u_x)
        k = self.to_k(u_y)
        v = self.to_v(u_y)
        q = q.view(batch_size, -1, self.n_heads, self.head_n_channels).permute(0, 2, 1, 3).contiguous()
        k = k.view(batch_size, -1, self.n_heads, self.head_n_channels).permute(0, 2, 1, 3).contiguous()
        v = v.view(batch_size, -1, self.n_heads, self.head_n_channels).permute(0, 2, 1, 3).contiguous()

        if weights is None:
            weights = torch.ones((u_y.shape[0], 1, u_y.shape[1], 1), device=u_y.device) / num_grid_points   # uniform weights
        else:
            weights = weights.view(batch_size, 1, num_grid_points, 1)

        # q = self.q_norm(q)
        k = self.normalize_wrt_domain(k, self.k_norm)
        v = self.normalize_wrt_domain(v, self.v_norm)

        if positional_embedding_module is not None:
            if self.pos_dim == 2:
                assert pos_x.shape[-1] == 2
                q_freqs_x = positional_embedding_module.forward(pos_x[..., 0], q.device)
                q_freqs_y = positional_embedding_module.forward(pos_x[..., 1], q.device)
                q_freqs_x = q_freqs_x.unsqueeze(1).repeat([1, self.n_heads, 1, 1])
                q_freqs_y = q_freqs_y.unsqueeze(1).repeat([1, self.n_heads, 1, 1])

                if pos_y is None:
                    k_freqs_x = q_freqs_x
                    k_freqs_y = q_freqs_y
                else:
                    k_freqs_x = positional_embedding_module.forward(pos_y[..., 0], k.device)
                    k_freqs_y = positional_embedding_module.forward(pos_y[..., 1], k.device)
                    k_freqs_x = k_freqs_x.unsqueeze(1).repeat([1, self.n_heads, 1, 1])
                    k_freqs_y = k_freqs_y.unsqueeze(1).repeat([1, self.n_heads, 1, 1])

                q = positional_embedding_module.apply_2d_rotary_positional_embedding_module(q, q_freqs_x, q_freqs_y)
                k = positional_embedding_module.apply_2d_rotary_positional_embedding_module(k, k_freqs_x, k_freqs_y)
            elif self.pos_dim == 1:
                assert pos_x.shape[-1] == 1

                q_freqs = positional_embedding_module.forward(pos_x[..., 0], q.device)
                q_freqs = q_freqs.unsqueeze(1).repeat([batch_size, self.n_heads, 1, 1])

                if pos_y is None:
                    k_freqs = q_freqs
                else:
                    k_freqs = positional_embedding_module.forward(pos_y[..., 0], k.device)
                    k_freqs = k_freqs.unsqueeze(1).repeat([batch_size, self.n_heads, 1, 1])

                q = positional_embedding_module.apply_rotary_pos_emb(q, q_freqs)
                k = positional_embedding_module.apply_rotary_pos_emb(k, k_freqs)
            else:
                raise Exception('Currently doesnt support relative embedding >= 3 dimensions')

        if associative:
            dots = torch.matmul(k.transpose(-1, -2), v)
            u = torch.matmul(q, dots) * weights
        else:
            # this is more efficient when num_grid_points<<channels
            kxy = torch.matmul(q, k.transpose(-1, -2))
            u = torch.matmul(kxy, v) * weights

        u = u.permute(0, 2, 1, 3).contiguous().view(batch_size, num_grid_points, self.n_heads*self.head_n_channels)
        u = self.to_out(u)
        if return_kernel:
            return u, kxy
        return u

