import torch
class Incremental():
    def __init__(self, model, incremental, incremental_loss_gap,
                incremental_resolution, incremental_eps, incremental_buffer,
                incremental_max_iter, incremental_grad_max_iter, 
                incremental_loss_eps, incremental_res_gap,
                dataset_resolution, dataset_sublist, dataset_indices, verbose=True) -> None:
        
        if incremental and incremental_loss_gap:
            raise ValueError(
                "Incremental and incremental loss gap cannot be used together")
            
        self.model = model
        self.ndim = len(model.n_modes)
        self.incremental_grad = incremental
        self.incremental_resolution = incremental_resolution
        self.incremental_loss_gap = incremental_loss_gap
        self.verbose = verbose

        if self.incremental_grad:
            # incremental gradient
            self.buffer = incremental_buffer
            self.grad_explained_ratio_threshold = incremental_eps
            self.max_iter = incremental_max_iter
            self.grad_max_iter =incremental_grad_max_iter

        if incremental_loss_gap:
            # incremental loss gap
            self.eps = incremental_loss_eps
            self.loss_list = []

        if incremental_resolution:
            # incremental resolution
            self.epoch_gap = incremental_res_gap
            self.indices = dataset_indices
            self.resolution = dataset_resolution
            self.sub_list = dataset_sublist

            self.subsammpling_rate = 1
            self.current_index = 0
            self.current_logged_epoch = 0
            self.current_sub = self.index_to_sub_from_table(self.current_index)
            self.current_res = self.sub_to_res(self.current_sub)
            
            if self.verbose:
                print(f'Incre Res Update: change index to {self.current_index}')
                print(f'Incre Res Update: change sub to {self.current_sub}')
                print(f'Incre Res Update: change res to {self.current_res}')
            
    # Algorithm 1: Incremental
    def loss_gap(self, loss):
        self.loss_list.append(loss)
        # method 1: loss_gap
        incremental_modes = self.model.fno_blocks.convs.incremental_n_modes[0]
        max_modes = self.model.fno_blocks.convs.n_modes[0]
        if len(self.loss_list) > 1:
            if abs(self.loss_list[-1] - self.loss_list[-2]) <= self.eps:
                if incremental_modes < max_modes:
                    incremental_modes += 1

        modes_list = tuple([incremental_modes] * self.ndim)
        self.model.fno_blocks.convs.incremental_n_modes = modes_list

    # Algorithm 2: Gradient based explained ratio
    def grad_explained(self):
        # for mode 1
        if not hasattr(self, 'accumulated_grad'):
            self.accumulated_grad = torch.zeros_like(
                self.model.fno_blocks.convs.weight[0])
        if not hasattr(self, 'grad_iter'):
            self.grad_iter = 1

        if self.grad_iter <= self.grad_max_iter:
            self.grad_iter += 1
            self.accumulated_grad += self.model.fno_blocks.convs.weight[0]
        else:
            incremental_final = []
            for i in range(self.ndim):
                max_modes = self.model.fno_blocks.convs.n_modes[i]
                incremental_modes = self.model.fno_blocks.convs.incremental_n_modes[i]
                weight = self.accumulated_grad
                strength_vector = []
                for mode_index in range(
                        min(weight.shape[1], incremental_modes)):
                    strength = torch.norm(
                        weight[:, mode_index, :], p='fro').cpu()
                    strength_vector.append(strength)
                expained_ratio = self.compute_explained_variance(
                    incremental_modes - self.buffer, torch.Tensor(strength_vector))
                if expained_ratio < self.grad_explained_ratio_threshold:
                    if incremental_modes < max_modes:
                        incremental_modes += 1
                incremental_final.append(incremental_modes)

            # update the modes and frequency dimensions
            self.grad_iter = 1
            self.accumulated_grad = torch.zeros_like(
                self.model.fno_blocks.convs.weight[0])
            self.model.fno_blocks.incremental_n_modes = tuple(incremental_final)

    # Algorithm 3: Regularize input resolution
    def incremental_resolution_regularize(self, x, y):
        return self.regularize_input_res(x, y)

    # Main step function: which algorithm to run
    def step(self, loss=None, epoch=None, x=None, y=None):
        if self.incremental_resolution and x is not None and y is not None:
            self.epoch_wise_res_increase(epoch)
            return self.incremental_resolution_regularize(x, y)
        if self.incremental_loss_gap and loss is not None:
            self.loss_gap(loss)
        if self.incremental_grad:
            self.grad_explained()
            
    def sub_to_res(self, sub):
        # Convert sub to resolution based
        return int(self.resolution / sub)

    def epoch_wise_res_increase(self, epoch):
        # Update the current_sub and current_res values based on the epoch
        if epoch % self.epoch_gap == 0 and epoch != 0 and (
                self.current_logged_epoch != epoch):
            self.current_index += 1
            self.current_sub = self.index_to_sub_from_table(self.current_index)
            self.current_res = self.sub_to_res(self.current_sub)
            self.current_logged_epoch = epoch

            if self.verbose:
                print(f'Incre Res Update: change index to {self.current_index}')
                print(f'Incre Res Update: change sub to {self.current_sub}')
                print(f'Incre Res Update: change res to {self.current_res}')

    def index_to_sub_from_table(self, index):
        # Get the sub value from the sub_list based on the index
        if index >= len(self.sub_list):
            return self.sub_list[-1]
        else:
            return self.sub_list[index]

    def compute_rank(self, tensor):
        # Compute the matrix rank of a tensor
        rank = torch.matrix_rank(tensor).cpu()
        return rank

    def compute_stable_rank(self, tensor):
        # Compute the stable rank of a tensor
        tensor = tensor.detach()
        fro_norm = torch.linalg.norm(tensor, ord='fro')**2
        l2_norm = torch.linalg.norm(tensor, ord=2)**2
        rank = fro_norm / l2_norm
        rank = rank.cpu()
        return rank

    def compute_explained_variance(self, frequency_max, s):
        # Compute the explained variance based on frequency_max and singular
        # values (s)
        s_current = s.clone()
        s_current[frequency_max:] = 0
        return 1 - torch.var(s - s_current) / torch.var(s)

    def regularize_input_res(self, x, y):
        # Regularize the input data based on the current_sub and dataset_name
        indices = torch.tensor(self.indices, device=x.device)
        x = torch.index_select(x, 0, index=indices)
        y = torch.index_select(y, 0, index=indices)
        return x, y