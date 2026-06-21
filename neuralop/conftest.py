import os

import torch

# Must happen before any FFT library initialises its thread pool.
# oneMKL creates DFTI plans whose configuration becomes inconsistent with
# autograd FFT backward passes when multiple threads are active on CPU.
if torch.backends.mkl.is_available():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
