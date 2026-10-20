import torch
import neuralop.mpu.comm as comm


def set_tf32(allow_tf32: bool):
    """Configure float32 math precision consistently across PyTorch versions.

    Parameters
    ----------
    allow_tf32 : bool
        Whether float32 matrix multiplications and cuDNN convolutions may use
        TensorFloat-32. ``False`` selects IEEE float32 math and is the safer
        choice for reproducible physics-informed training.

    Notes
    -----
    PyTorch 2.9 introduced the ``torch.backends.fp32_precision`` API and is
    deprecating the older ``allow_tf32`` flags. Older PyTorch releases use
    the legacy flags, so this helper keeps the training scripts compatible
    with both APIs without mixing them on newer releases.
    """
    if not isinstance(allow_tf32, bool):
        raise TypeError(f"allow_tf32 must be a bool, got {type(allow_tf32).__name__}")

    precision = "tf32" if allow_tf32 else "ieee"

    if hasattr(torch.backends, "fp32_precision"):
        torch.backends.fp32_precision = precision
        torch.backends.cuda.matmul.fp32_precision = precision
        torch.backends.cudnn.fp32_precision = precision
        torch.backends.cudnn.conv.fp32_precision = precision
        torch.backends.cudnn.rnn.fp32_precision = precision
        return

    try:
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    except AttributeError:
        pass

    cuda_backends = getattr(torch.backends, "cuda", None)
    cuda_matmul = getattr(cuda_backends, "matmul", None)
    if cuda_matmul is not None:
        cuda_matmul.allow_tf32 = allow_tf32

    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None:
        cudnn.allow_tf32 = allow_tf32


def setup(config):
    """A convenience function to intialize the device, setup torch settings and
    check multi-grid and other values. It sets up distributed communitation, if used.

    Parameters
    ----------
    config : dict
        this function checks:
        * config.distributed (use_distributed, seed)
        * config.data (n_train, batch_size, test_batch_sizes, n_tests, test_resolutions)
        * config.opt (allow_tf32)

    Returns
    -------
    device, is_logger
        device : torch.device
        is_logger : bool
    """
    seed = config.distributed.seed

    if config.distributed.use_distributed:
        comm.init(
            model_parallel_size=config.distributed.model_parallel_size,
            verbose=config.verbose,
        )

        # Set process 0 to log screen and wandb
        is_logger = comm.get_local_rank() == 0

        # Set device and random seed
        device = torch.device(f"cuda:{comm.get_local_rank()}")

        if seed is not None:
            seed = seed + comm.get_data_parallel_rank()

        # Ensure batch can be evenly split among the model-parallel group
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
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    # Set device, random seed and optimization
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)

        if seed is not None:
            torch.cuda.manual_seed(seed)
        increase_l2_fetch_granularity()
        set_tf32(config.opt.get("allow_tf32", True))

        torch.backends.cudnn.benchmark = True

    if seed is not None:
        torch.manual_seed(seed)

    return device, is_logger


def increase_l2_fetch_granularity():
    try:
        import ctypes

        _libcudart = ctypes.CDLL("libcudart.so")
        # Set device limit on the current device
        # cudaLimitMaxL2FetchGranularity = 0x05
        pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
        _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
        _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
        assert pValue.contents.value == 128
    except:
        return
