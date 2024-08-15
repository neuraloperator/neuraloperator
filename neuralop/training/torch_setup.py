from typing import Iterable

import torch
from torch import nn
import neuralop.mpu.comm as comm


def setup(config):
    """A convenience function to intialize the device, setup torch settings and
    check multi-grid and other values. It sets up distributed communitation, if used.
    
    Parameters
    ----------
    config : dict 
        this function checks:
        * config.distributed (use_distributed, seed)
        * config.data (n_train, batch_size, test_batch_sizes, n_tests, test_resolutions)
    
    Returns
    -------
    device, is_logger
        device : torch.device
        is_logger : bool
    """
    if config.distributed.use_distributed:
        comm.init(config, verbose=config.verbose)

        #Set process 0 to log screen and wandb
        is_logger = (comm.get_world_rank() == 0)

        #Set device and random seed
        device = torch.device(f"cuda:{comm.get_local_rank()}")
        seed = config.distributed.seed + comm.get_data_parallel_rank()

        #Ensure every iteration has the same amount of data 
        assert(config.data.n_train % config.data.batch_size == 0), (
            f'The number of training samples={config.data.n_train} cannot be divided by the batch_size={config.data.batch_size}.'
        )
        for j in range(len(config.data.test_batch_sizes)):
            assert(config.data.n_tests[j] % config.data.test_batch_sizes[j] == 0), (
                f'The number of training samples={config.data.n_tests[j]}'
                f' cannot be divided by the batch_size={config.data.test_batch_sizes[j]}'
                f' for test resolution {config.data.test_resolutions[j]}.'
            )

        #Ensure batch can be evenly split among the data-parallel group
        #NOTE: Distributed sampler NOT implemented: set model_parallel_size = # of GPUS
        assert (config.data.batch_size % comm.get_data_parallel_size() == 0), (
                f'Batch of size {config.data.batch_size} can be evenly split among the data-parallel group={comm.get_data_parallel_size()}.'
        )
        config.data.batch_size = config.data.batch_size // comm.get_data_parallel_size()

        #Ensure batch can be evenly split among the model-parallel group
        if config.patching.levels > 0:
            assert(config.data.batch_size*(2**(2*config.patching.levels)) % comm.get_model_parallel_size() == 0), (
                f'With MG patching, total batch-size of {config.data.batch_size*(2**(2*config.patching.levels))}'
                f' ({config.data.batch_size} times {(2**(2*config.patching.levels))}).'
                f' However, this total batch-size cannot be evenly split among the {comm.get_model_parallel_size()} model-parallel groups.'
            )
            for b_size in config.data.test_batch_sizes:
                assert (b_size*(2**(2*config.patching.levels)) % comm.get_model_parallel_size() == 0), (
                f'With MG patching, for test resolution of {config.data.test_resolutions[j]}'
                f' the total batch-size is {config.data.batch_size*(2**(2*config.patching.levels))}'
                f' ({config.data.batch_size} times {(2**(2*config.patching.levels))}).'
                f' However, this total batch-size cannot be evenly split among the {comm.get_model_parallel_size()} model-parallel groups.'
                )

    else:
        is_logger = True
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        if 'seed' in config.distributed:
            seed = config.distributed.seed

    #Set device, random seed and optimization
    if torch.cuda.is_available():

        torch.cuda.set_device(device.index)

        if 'seed' in config.distributed:
            torch.cuda.manual_seed(seed)
        increase_l2_fetch_granularity()
        try:
            torch.set_float32_matmul_precision('high')
        except AttributeError:
            pass
        
        torch.backends.cudnn.benchmark = True

    if 'seed' in config.distributed:
        torch.manual_seed(seed)

    return device, is_logger

def get_optimizer(
    name: str,
    parameters: Iterable[nn.Parameter],
    learning_rate: float,
    weight_decay: float,
    momentum: float=1e-4,
    nesterov: bool=False,
):
    """get_optimizer returns an optimizer configured according
    to a neuraloperator training script's config file.

    Parameters
    ----------
    name : str
        name of optimizer, either 'Adam' or 'SGD'
    parameters : Iterable[nn.Parameter]
        parameters to optimize
    learning_rate : float
        optimizer's LR
    weight_decay : float
        optimizer's weight decay
    momentum : float, optional
        momentum param if optimizer is 'SGD'
    nesterov: bool, optional
        whether to use nesterov momentum if optimizer is 'SGD'
    """
    if name == 'Adam':
        optimizer = torch.optim.Adam(
            params=parameters,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
    elif name == 'SGD':
        optimizer = torch.optim.SGD(
            params=parameters,
            lr=learning_rate,
            momentum=momentum,
            nesterov=nesterov
        )
    return optimizer

def get_scheduler(name: str,
                  optimizer: torch.optim.Optimizer,
                  gamma: float=None,
                  patience: int=None,
                  T_max: int=None,
                  step_size: int=None):
    """get_scheduler returns a learning rate scheduler configured according
    to a neuraloperator training script's config file.

    Parameters
    ----------
    name : str
        name of LR Scheduler module.
        Either 'StepLR', 'ReduceLROnPlateau', or 'CosineAnnealingLR'
    optimizer : torch.optim.Optimizer
        optimizer to schedule
    gamma : float, optional
        LR scaling factor for use in some schedulers, by default None
    patience : int, optional
        number of epochs to wait to drop LR in some schedulers, by default None
    T_max : int, optional
        max number of epochs for some schedulers, by default None
    step_size : int, optional
        number of epochs per step in StepLR, by default None

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler
        module to use to schedule LR in training
    """

    if name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=gamma,
            patience=patience,
            mode="min",
        )
    elif name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max
        )
    elif name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    else:
        raise ValueError(f"Got scheduler={name}, expected one of \'StepLR\', \'ReduceLROnPlateau\', \'CosineAnnealingLR\'")

    return scheduler


def increase_l2_fetch_granularity():
    try:
        import ctypes

        _libcudart = ctypes.CDLL('libcudart.so')
        # Set device limit on the current device
        # cudaLimitMaxL2FetchGranularity = 0x05
        pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
        _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
        _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
        assert pValue.contents.value == 128
    except:
        return
