from typing import Optional
from zencfg import ConfigBase

class WandbConfig(ConfigBase):
    log: bool = False
    entity: Optional[str] = None
    project: Optional[str] = None
    name: Optional[str] = None
    group: str = None
    sweep: bool = False
    log_output: bool = True