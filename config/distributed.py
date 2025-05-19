from typing import List, Any, Optional

from zencfg import ConfigBase

class DistributedConfig(ConfigBase):
    use_distributed: bool = False
    model_parallel_size: Optional[int] = 1
    seed: Optional[int] = None
