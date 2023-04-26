import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

class Paramaters:
    def __init__(self, model, incremental, incremental_loss_gap, incremental_resolution, dataset_name) -> None:
        self.model = model
        self.ndim = len(model.n_modes)
        self.incremental_grad = incremental
        self.incremental_resolution = incremental_resolution
        self.incremental_loss_gap = incremental_loss_gap
        self.dataset_name = dataset_name
        
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
            self.epoch_gap = 100
            if self.dataset_name == 'SmallDarcy':
                self.sub_list = [16, 8, 4, 2, 1]
            if self.dataset_name == 'Darcy':
                self.sub_list = [10,8,4,2,1]
            elif self.dataset_name == "Burgers":
                self.sub_list = [256,64,16,8,1]
            elif self.dataset_name == "NavierStokes":
                self.sub_list = [32,16,8,4,1]
            elif self.dataset_name == "Vorticity":
                self.sub_list = [128,64,32,16,1]
                
            self.subsammpling_rate = 1   
            self.current_index = 0
            self.current_logged_epoch = 0
            self.current_sub = self.index_to_sub_from_table(self.current_index)
            self.current_res = self.sub_to_res(self.current_sub)

            print(f'Incre Res Update: change index to {self.current_index}')
            print(f'Incre Res Update: change sub to {self.current_sub}')
            print(f'Incre Res Update: change res to {self.current_res}')
        
    def sub_to_res(self, sub):
        if self.dataset_name == 'SmallDarcy':
            return self.small_darcy_sub_to_res(sub)
        if self.dataset_name == 'Burgers':
            return self.burger_sub_to_res(sub)
        elif self.dataset_name == 'Darcy':
            return self.darcy_sub_to_res(sub)
        elif self.dataset_name == 'NavierStokes':
            return self.navier_sub_to_res(sub)
        elif self.dataset_name == 'NavierStokesHighFrequency':
            return self.navier_high_sub_to_res(sub)
              
    def burger_sub_to_res(self, sub):
        return int(2**13 / sub)

    def small_darcy_sub_to_res(self, sub):
        return int(16 / sub)

    def darcy_sub_to_res(self, sub):
        return int(((241 - 1)/sub) + 1)

    def navier_sub_to_res(self, sub):
        return 64 // sub

    def navier_high_sub_to_res(self, sub):
        return 256 // sub

    def epoch_wise_res_increase(self, epoch):
        if epoch % self.epoch_gap == 0 and epoch != 0 and (self.current_logged_epoch < epoch):
            self.current_index += 1
            self.current_sub = self.index_to_sub_from_table(self.current_index)
            self.current_res = self.sub_to_res(self.current_sub)
            self.current_logged_epoch = epoch

            print(f'Incre Res Update: change index to {self.current_index}')
            print(f'Incre Res Update: change sub to {self.current_sub}')
            print(f'Incre Res Update: change res to {self.current_res}')
            
    def index_to_sub_from_table(self, index):
        if index >= len(self.sub_list):
            return self.sub_list[-1]
        else:
            return self.sub_list[index]
                
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

    def regularize_input_res(self, x, y):
        if self.dataset_name == 'Burgers':
            x = x[:, ::self.current_sub]
            y = y[:, ::self.current_sub]
        elif self.dataset_name == 'Darcy':
            x = x[:, :, ::self.current_sub, ::self.current_sub]
            y = y[:, :, ::self.current_sub, ::self.current_sub]
        elif self.dataset_name == 'SmallDarcy':
            x = x[::self.current_sub, :, :, :]
            y = y[::self.current_sub, :, :, :]
        elif self.dataset_name == 'NavierStokes':
            T_in = self._datamodule.T_in
            T = self._datamodule.T
            x = x[:, ::self.current_sub, ::self.current_sub, :]
            y = y[:, ::self.current_sub, ::self.current_sub, :]
        elif self.dataset_name == 'NavierStokesHighFrequency':
            T_in = self._datamodule.T_in
            T = self._datamodule.T
            x = x[:, ::self.current_sub, ::self.current_sub]
            y = y[:, ::self.current_sub, ::self.current_sub]
        return x,y