import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

def compute_rank(tensor):
    rank = torch.matrix_rank(tensor).cpu()
    return rank
    
def compute_stable_rank(tensor):
    # tensor = tensor.detach().cpu()
    tensor = tensor.detach()
    fro_norm = torch.linalg.norm(tensor, ord='fro')**2
    l2_norm = torch.linalg.norm(tensor, ord=2)**2
    rank = fro_norm / l2_norm
    rank = rank.cpu()
    return rank

def compute_explained_variance(frequency_max, s):
	s_current = s.clone()
	s_current[frequency_max:] = 0
	return 1 - torch.var(s - s_current) / torch.var(s)

def loss_gap(eps, max_modes, incremental_modes, loss_list):
    # method 1: loss_gap
    if len(loss_list) > 1:
        if abs(loss_list[-1] - loss_list[-2]) <= eps:
            if incremental_modes < max_modes:
                incremental_modes += 1

    return incremental_modes

def grad_explained(weights, grad_explained_ratio_threshold, max_modes, incremental_modes, buffer):        
    # for mode 1
    strength_vector = []
    for mode_index in range(incremental_modes):
        strength = torch.norm(weights[:,mode_index,:], p='fro').cpu()
        strength_vector.append(strength)
    expained_ratio = compute_explained_variance(incremental_modes - buffer, torch.Tensor(strength_vector))
    if expained_ratio < grad_explained_ratio_threshold:
        if incremental_modes < max_modes:
            incremental_modes += 1

    return incremental_modes