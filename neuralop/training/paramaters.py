import torch
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig


class Paramaters:
    def __init__(
            self,
            model,
            incremental,
            incremental_loss_gap,
            incremental_resolution,
            dataset_name) -> None:
        self.model = model
        self.ndim = len(model.n_modes)
        self.incremental_grad = incremental
        self.incremental_resolution = incremental_resolution
        self.incremental_loss_gap = incremental_loss_gap
        self.dataset_name = dataset_name

        # Initialize ConfigPipeline to read configurations from YAML and
        # command line arguments
        pipe = ConfigPipeline(
            [
                YamlConfig(
                    # Add the config path to the incremental config file
                    #'/workspace/markov_neural_operator/scripts/config/incremental.yaml',
                    '/home/user/markov_neural_operator/scripts/config/incremental.yaml',
                    config_name='default'),
                ArgparseConfig(
                    infer_types=True,
                    config_name=None,
                    config_file=None)])
        config = pipe.read_conf()
        paramaters = config.incremental
                
        if self.incremental_grad:
            # incremental gradient
            paramaters_grad = paramaters.incremental_grad
            self.buffer = paramaters_grad.buffer_modes
            self.grad_explained_ratio_threshold = paramaters_grad.grad_explained_ratio_threshold
            self.max_iter = paramaters_grad.max_iter
            self.grad_max_iter = paramaters_grad.grad_max_iter

        if self.incremental_loss_gap:
            # incremental loss gap
            paramaters_gap = 0.01 #paramaters.incremental_loss_gap
            self.eps = 0.01 #paramaters_gap.eps
            self.loss_list = []

        if self.incremental_resolution:
            # incremental resolution
            paramaters_resolution = paramaters.incremental_resolution
            self.epoch_gap = 10 #paramaters_resolution.epoch_gap
            self.new_index = config.checkpoint.sub_list_index

            # Determine the sub_list based on the dataset_name
            # Best to do powers of two as FFT is faster and FFTW++ library
            # optimizes for powers of two
            if self.dataset_name == 'SmallDarcy':
                self.sub_list = paramaters.dataset.SmallDarcy  # cannot do incremental resolution
            elif self.dataset_name == 'Darcy' or self.dataset_name=='Re5000':
                self.sub_list = [4, 2, 1] #paramaters.dataset.Darcy
            elif self.dataset_name == "Burgers":
                self.sub_list = paramaters.dataset.Burgers
            elif self.dataset_name == "NavierStokes":
                self.sub_list = paramaters.dataset.NavierStokes
            elif self.dataset_name == "Vorticity":
                self.sub_list = paramaters.dataset.Voriticity

            self.subsammpling_rate = 1
            self.current_index = self.new_index
            self.current_logged_epoch = 0
            self.current_sub = self.index_to_sub_from_table(self.current_index)
            self.current_res = self.sub_to_res(self.current_sub)

            print(f'Incre Res Update: change index to {self.current_index}')
            print(f'Incre Res Update: change sub to {self.current_sub}')
            print(f'Incre Res Update: change res to {self.current_res}')

    def sub_to_res(self, sub):
        # Convert sub to resolution based on the dataset_name
        if self.dataset_name == 'SmallDarcy':
            return self.small_darcy_sub_to_res(sub)
        if self.dataset_name == 'Burgers':
            return self.burger_sub_to_res(sub)
        elif self.dataset_name == 'Darcy' or self.dataset_name=='Re5000':
            return self.darcy_sub_to_res(sub)
        elif self.dataset_name == 'NavierStokes':
            return self.navier_sub_to_res(sub)
        elif self.dataset_name == 'NavierStokesHighFrequency':
            return self.navier_high_sub_to_res(sub)

    def burger_sub_to_res(self, sub):
        # Calculate resolution based on sub for the Burgers dataset
        return int(2**13 / sub)

    def small_darcy_sub_to_res(self, sub):
        # Calculate resolution based on sub for the SmallDarcy dataset
        return int(16 / sub)

    def darcy_sub_to_res(self, sub):
        # Calculate resolution based on sub for the Darcy dataset
        return int((128 / sub))

    def navier_sub_to_res(self, sub, resolution=512):
        # Calculate resolution based on sub for the NavierStokes dataset
        # Assumes one is using the default high resolution dataset
        return resolution // sub

    def navier_high_sub_to_res(self, sub):
        # Calculate resolution based on sub for the NavierStokesHighFrequency
        # dataset
        return 256 // sub

    def epoch_wise_res_increase(self, epoch):
        # Update the current_sub and current_res values based on the epoch
        if epoch % self.epoch_gap == 0 and epoch != 0 and (
                self.current_logged_epoch != epoch):
            self.current_index += 1
            self.current_sub = self.index_to_sub_from_table(self.current_index)
            self.current_res = self.sub_to_res(self.current_sub)
            self.current_logged_epoch = epoch
            self.new_index += 1

            print(f'Incre Res Update: change index to {self.current_index}')
            print(f'Incre Res Update: change sub to {self.current_sub}')
            print(f'Incre Res Update: change res to {self.current_res}')

    def index_to_sub_from_table(self, index):
        # Get the sub value from the sub_list based on the index
        if index >= len(self.sub_list):
            return int(self.sub_list[-1])
        else:
            """if index == 3 and self.current_logged_epoch > 480:
                print("Now only running on the highest resolution")
                return int(self.sub_list[3])
            else:
                if index != 3:
                    print("Running as usual")
                    return int(self.sub_list[index])
                else:
                    print("Running on 2nd lowest resolution")"""
            return int(self.sub_list[index])

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
        if self.dataset_name == 'Burgers':
            x = x[:, :, ::self.current_sub]
            y = y[:, ::self.current_sub]
        elif self.dataset_name == 'SmallDarcy':
            x = x[:, :, ::self.current_sub, ::self.current_sub]
            y = y[:, ::self.current_sub, ::self.current_sub]
        elif self.dataset_name == 'Darcy' or self.dataset_name=='Re5000':
            batchsize, channels, *mode_sizes = x.shape
            s1 = (128 // self.current_sub)
            mode_sizes[-1] = s1
            mode_sizes[-2] = s1
            """if s1 == 128 or s1 == 64:
                x = torch.fft.rfftn(x.float(), norm="backward", dim=[-2, -1])
                x = torch.fft.irfftn(x, s=(mode_sizes), norm="backward")
                y = torch.fft.rfftn(y.float(), norm="backward", dim=[-2, -1])
                y = torch.fft.irfftn(y, s=(mode_sizes), norm="backward")
            else:"""
            x = x[:, :, ::self.current_sub, ::self.current_sub]
            y = y[:, :, ::self.current_sub, ::self.current_sub]
        elif self.dataset_name == 'NavierStokes':
            x = x[:, :, ::self.current_sub, ::self.current_sub]
            y = y[:, :, ::self.current_sub, ::self.current_sub]
        elif self.dataset_name == 'NavierStokesHighFrequency':
            x = x[:, ::self.current_sub, ::self.current_sub]
            y = y[:, ::self.current_sub, ::self.current_sub]
        return x, y, self.current_sub