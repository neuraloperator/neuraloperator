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

import torch
import torch.nn.functional as F
import torch.distributed as dist


def get_memory_format(tensor):
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    else:
        return torch.contiguous_format


def pad_helper(tensor, dim, new_size, mode="zero"):
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    ndim_pad = ndim - dim
    output_shape = [0 for _ in range(2*ndim_pad)]
    orig_size = tensor.shape[dim]
    output_shape[1] = new_size - orig_size
    tensor_pad = F.pad(tensor, output_shape, mode='constant', value=0.)
    
    if mode == "conj":
        lhs_slice = [slice(0,x) if idx != dim else slice(orig_size, new_size) for idx,x in enumerate(tensor.shape)]
        rhs_slice = [slice(0,x) if idx != dim else slice(1, output_shape[1]+1) for idx,x in enumerate(tensor.shape)]
        tensor_pad[lhs_slice] = torch.flip(torch.conj(tensor_pad[rhs_slice]), dims=[dim])
        
    return tensor_pad


def truncate_helper(tensor, dim, new_size):
    input_format = get_memory_format(tensor)
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    output_slice = [slice(0,x) if idx != dim else slice(0,new_size) for idx,x in enumerate(tensor.shape)]
    tensor_trunc = tensor[output_slice].contiguous(memory_format=input_format)
    
    return tensor_trunc


def split_tensor_along_dim(tensor, dim, num_chunks):
    assert dim < tensor.dim(), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"
    assert (tensor.shape[dim] % num_chunks == 0), f"Error, cannot split dim {dim} evenly. Dim size is \
                                                  {tensor.shape[dim]} and requested numnber of splits is {num_chunks}"
    chunk_size = tensor.shape[dim] // num_chunks
    tensor_list = torch.split(tensor, chunk_size, dim=dim)
    
    return tensor_list


# distributed primitives
def _transpose(tensor, dim0, dim1, group=None, async_op=False):
    # get input format
    input_format = get_memory_format(tensor)
    
    # get comm params
    comm_size = dist.get_world_size(group=group)

    # split and local transposition
    split_size = tensor.shape[dim0] // comm_size
    x_send = [y.contiguous(memory_format=input_format) for y in torch.split(tensor, split_size, dim=dim0)]
    x_recv = [torch.empty_like(x_send[0]) for _ in range(comm_size)]
    
    # global transposition
    req = dist.all_to_all(x_recv, x_send, group=group, async_op=async_op)
    
    return x_recv, req 


def _reduce(input_, use_fp32=True, group=None):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_
    
    # All-reduce.
    if use_fp32:
        dtype = input_.dtype
        inputf_ = input_.float()
        dist.all_reduce(inputf_, group=group)
        input_ = inputf_.to(dtype)
    else:
        dist.all_reduce(input_, group=group)
        
    return input_


def _split(input_, dim_, group=None):
    """Split the tensor along its last dimension and keep the corresponding slice."""
    # get input format
    input_format = get_memory_format(input_)
    
    # Bypass the function if we are using only 1 GPU.
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_
    
    # Split along last dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size)
    
    # Note: torch.split does not create contiguous tensors by default.
    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous(memory_format=input_format)
    
    return output


def _gather(input_, dim_, group=None):
    """Gather tensors and concatinate along the last dimension."""
    # get input format
    input_format = get_memory_format(input_) 

    comm_size = dist.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if comm_size==1:
        return input_

    # sanity checks
    assert(dim_ < input_.dim()), f"Error, cannot gather along {dim_} for tensor with {input_.dim()} dimensions."

    # Size and dimension.
    comm_rank = dist.get_rank(group=group)
    
    tensor_list = [torch.empty_like(input_) for _ in range(comm_size)]
    tensor_list[comm_rank] = input_.contiguous(memory_format=input_format)
    dist.all_gather(tensor_list, input_, group=group)
    
    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim_).contiguous(memory_format=input_format)
    
    return output