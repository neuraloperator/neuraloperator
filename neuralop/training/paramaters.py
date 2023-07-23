import torch
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import os

class Paramaters:
    def __init__(
            self,
            model,
            incremental,
            incremental_loss_gap,
            incremental_resolution) -> None:
        self.model = model
        self.ndim = len(model.n_modes)
        self.incremental_grad = incremental
        self.incremental_resolution = incremental_resolution
        self.incremental_loss_gap = incremental_loss_gap
        # Get the user's home directory path
        home_dir = os.path.expanduser("~")

        # Construct the full path by joining the home directory path with the relative path
        config_path = os.path.join(home_dir, "neuraloperator", "config", "incremental.yaml")
        
        # Initialize ConfigPipeline to read configurations from YAML and
        # command line arguments
        pipe = ConfigPipeline(
            [
                YamlConfig(
                    # Add the config path to the incremental config file
                    config_file=config_path,
                    config_name='default'),
                ArgparseConfig(
                    infer_types=True,
                    config_name=None,
                    config_file=None)])
        config = pipe.read_conf()
        paramaters = config.incremental
        self.dataset_name = paramaters.dataset_name

        if self.incremental_grad:
            # incremental gradient
            paramaters_grad = paramaters.incremental_grad
            self.buffer = paramaters_grad.buffer_modes
            self.grad_explained_ratio_threshold = paramaters_grad.grad_explained_ratio_threshold
            self.max_iter = paramaters_grad.max_iter
            self.grad_max_iter = paramaters_grad.grad_max_iter

        if self.incremental_loss_gap:
            # incremental loss gap
            paramaters_gap = paramaters.incremental_loss_gap
            self.eps = paramaters_gap.eps
            self.loss_list = []

        if self.incremental_resolution:
            # incremental resolution
            paramaters_resolution = paramaters.incremental_resolution
            self.epoch_gap = paramaters_resolution.epoch_gap
            self.indices = paramaters.dataset_indices
            self.resolution = paramaters.dataset_resolution
            self.sub_list = paramaters.dataset_sub_list
            
            self.subsammpling_rate = 1
            self.current_index = 0
            self.current_logged_epoch = 0
            self.current_sub = self.index_to_sub_from_table(self.current_index)
            self.current_res = self.sub_to_res(self.current_sub)

            print(f'Incre Res Update: change index to {self.current_index}')
            print(f'Incre Res Update: change sub to {self.current_sub}')
            print(f'Incre Res Update: change res to {self.current_res}')

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
