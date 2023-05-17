import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
from .paramaters import Paramaters

class Incremental(Paramaters):
    
    def __init__(self, model, incremental, incremental_loss_gap, incremental_resolution, dataset_name) -> None:
        super().__init__(model, incremental, incremental_loss_gap, incremental_resolution, dataset_name)

        if self.incremental_grad and self.incremental_loss_gap:
            raise ValueError("Incremental and incremental loss gap cannot be used together")

    # Algorithm 1: Incremental        
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

    # Algorithm 2: Gradient based explained ratio
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
    
    # Algorithm 3: Regularize input resolution
    def incremental_resolution_regularize(self, x, y):
        return self.regularize_input_res(x, y)
    
    def step(self, loss = None, epoch = None, x = None, y = None):
        if self.incremental_resolution and x != None and y != None:
            self.epoch_wise_res_increase(epoch)
            return self.incremental_resolution_regularize(x, y)
        if self.incremental_loss_gap and loss != None:
            self.loss_gap(loss)
        if self.incremental_grad:
            self.grad_explained()
    