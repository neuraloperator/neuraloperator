from typing import List, Any, Optional

from zencfg import ConfigBase

class DistributedConfig(ConfigBase):
    """
    DistributedConfig provides config options for multi-GPU 
    and multi-node settings. Our current setup depends on ``torchrun``-based
    Elastic Launch and the ``nccl`` distributed backend for communication. 

    Parameters
    ----------
    use_distributed: bool, default False
        Whether to use distributed data/model parallelism
    model_parallel_size: Optional[int], default 1
        number of GPUs across which to spread model layers,
        by default 1. If 1, does not perform **any** model parallelism.
    seed: Optional[int] = None
        special distributed random torch seed for reproducibility.
    """
    use_distributed: bool = False
    model_parallel_size: Optional[int] = 1
    seed: Optional[int] = None
