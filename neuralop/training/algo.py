import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

class Incremental:
    def __init__(self, model, incremental, incremental_loss_gap, incremental_resolution) -> None:
        self.model = model
        self.ndim = len(model.n_modes)
        self.incremental_grad = incremental
        self.incremental_resolution = incremental_resolution
        self.incremental_loss_gap = incremental_loss_gap
        
        if self.incremental_grad:
            # incremental
            self.buffer = 5
            self.grad_explained_ratio_threshold = 0.99
            self.max_iter = 1
            self.grad_max_iter = 10
        
        if self.incremental_loss_gap:
            # loss gap
            self.eps = 1e-2
            self.loss_list = []
            
        if self.incremental_resolution:
            pass

    def loss_gap(self, loss):
        self.loss_list.append(loss)
        # method 1: loss_gap
        incremental_modes = self.model.convs.incremental_n_modes[0]
        max_modes = self.model.convs.n_modes[0]
        if len(self.loss_list) > 1:
            if abs(self.loss_list[-1] - self.loss_list[-2]) <= self.eps:
                if incremental_modes < max_modes:
                    incremental_modes += 1
    
        modes_list = tuple([incremental_modes] * self.ndim)
        self.model.incremental_n_modes = modes_list

    def grad_explained(self):        
        # for mode 1
        if not hasattr(self, 'accumulated_grad'):
            self.accumulated_grad = torch.zeros_like(self.model.convs.weight[0])
        if not hasattr(self, 'grad_iter'):
            self.grad_iter = 1 
    
        if self.grad_iter <= self.grad_max_iter:
            self.grad_iter += 1
            self.accumulated_grad += self.model.convs.weight[0]
        else:
            # weights, grad_explained_ratio_threshold, max_modes, incremental_modes, buffer
            # loop over all frequency dimensions
            # create a list of eventual modes
            incremental_final = []
            for i in range(self.ndim):
                max_modes = self.model.convs.n_modes[i]
                incremental_modes = self.model.convs.incremental_n_modes[i]
                weight = self.accumulated_grad
                strength_vector = []
                for mode_index in range(incremental_modes):
                    strength = torch.norm(weight[:,mode_index,:], p='fro').cpu()
                    strength_vector.append(strength)
                expained_ratio = self.compute_explained_variance(incremental_modes - self.buffer, torch.Tensor(strength_vector))
                if expained_ratio < self.grad_explained_ratio_threshold:
                    if incremental_modes < max_modes:
                        incremental_modes += 1
                incremental_final.append(incremental_modes)
            
            # update the modes and frequency dimensions
            self.grad_iter = 1
            self.accumulated_grad = torch.zeros_like(self.model.convs.weight[0])
            self.model.incremental_n_modes = tuple(incremental_final)

    def step(self, loss = False):
        if self.incremental_loss_gap:
            self.loss_gap(loss)
        if self.incremental_grad:
            self.grad_explained()
        if self.incremental_resolution:
            self.incremental_resolution()
    
    def compute_rank(self, tensor):
        rank = torch.matrix_rank(tensor).cpu()
        return rank
        
    def compute_stable_rank(self, tensor):
        # tensor = tensor.detach().cpu()
        tensor = tensor.detach()
        fro_norm = torch.linalg.norm(tensor, ord='fro')**2
        l2_norm = torch.linalg.norm(tensor, ord=2)**2
        rank = fro_norm / l2_norm
        rank = rank.cpu()
        return rank

    def compute_explained_variance(self, frequency_max, s):
        s_current = s.clone()
        s_current[frequency_max:] = 0
        return 1 - torch.var(s - s_current) / torch.var(s)

    def incremental_resolution():
        pass