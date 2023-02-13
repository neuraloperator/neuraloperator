# coding=utf-8
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import types
from typing import Any

import torch
import torch.distributed as dist
from .comm import get_model_parallel_group

# torch utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# helper functions
from .helpers import split_tensor_along_dim
from .helpers import _reduce
from .helpers import _split
from .helpers import _gather

# model parallel
class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_
    
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, group=get_model_parallel_group())


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""
    
    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_, group=get_model_parallel_group())
    
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_, group=get_model_parallel_group())
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""
    
    @staticmethod
    def symbolic(graph, input_, dim_):
        return _split(input_, dim_, group=get_model_parallel_group())
    
    @staticmethod
    def forward(ctx, input_, dim_):
        ctx.dim = dim_
        return _split(input_, dim_, group=get_model_parallel_group())
    
    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, ctx.dim, group=get_model_parallel_group()), None
    
    
class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""
    
    @staticmethod
    def symbolic(graph, input_, dim_):
        return _gather(input_, dim_, group=get_model_parallel_group())
    
    @staticmethod
    def forward(ctx, input_, dim_):
        ctx.dim = dim_
        return _gather(input_, dim_, group=get_model_parallel_group())
    
    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.dim, group=get_model_parallel_group()), None
    
# -----------------
# Helper functions.
# -----------------
# matmul parallel
def copy_to_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_model_parallel_region(input_, dim):
    return _ScatterToModelParallelRegion.apply(input_, dim)


def gather_from_model_parallel_region(input_, dim):
    return _GatherFromModelParallelRegion.apply(input_, dim)