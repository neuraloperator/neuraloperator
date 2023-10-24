import os

import torch
import torch.distributed as dist
import os
import logging
import torch
import torch.distributed as dist
import datetime as dt

class disable_logging(object):
    def __init__(self, level=logging.ERROR):
        logging.disable(level=level)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        logging.disable(level=logging.NOTSET)


# dummy placeholders
_DATA_PARALLEL_GROUP = None
_MODEL_PARALLEL_GROUP = None

# world comm
def get_world_size():
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()


def get_world_rank():
    if not dist.is_initialized():
        return 0
    else:
        return dist.get_rank()


def get_local_rank():
    if not dist.is_initialized():
        return 0
    else:
        return get_world_rank() % torch.cuda.device_count()


# data parallel
def get_data_parallel_size():
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size(group=_DATA_PARALLEL_GROUP)


def get_data_parallel_rank():
    if not dist.is_initialized():
        return 0
    else:
        return dist.get_rank(group=_DATA_PARALLEL_GROUP)


def get_data_parallel_group():
    assert dist.is_initialized(), "Error, initialize torch.distributed first"
    return _DATA_PARALLEL_GROUP 


# model parallel
def get_model_parallel_size():
    if not dist.is_initialized() or (_MODEL_PARALLEL_GROUP is None):
        return 1
    else:
        return dist.get_world_size(group=_MODEL_PARALLEL_GROUP)


def get_model_parallel_rank():
    if not dist.is_initialized() or (_MODEL_PARALLEL_GROUP is None):
        return 0
    else:
        return dist.get_rank(group=_MODEL_PARALLEL_GROUP)


def get_model_parallel_group():
    assert dist.is_initialized(), "Error, initialize torch.distributed first"
    return _MODEL_PARALLEL_GROUP  

def init(world_size, world_rank, config):
    """
    Initializes torch.distributed process group via pure Python using 
    the nccl backend and torch.multiprocessing.spawn
    """
    master_addr = 'localhost'
    master_port = '12355'

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port  
    
    if config.distributed.wireup_store == "file":

        wireup_file_path = os.getenv('WIREUP_FILE_PATH')
        wireup_store = dist.FileStore(wireup_file_path, world_size)
    
    elif config.distributed.wireup_store == "tcp":
        # create tcp store
        wireup_store = dist.TCPStore(host_name = master_addr,
                                        port = master_port,
                                        world_size = world_size,
                                        is_master = (world_rank == 0),
                                        timeout = dt.timedelta(seconds=900))
    
    # initialize process group
    dist.init_process_group(backend='nccl', 
                            world_size=world_size, 
                            rank=world_rank,
                            store=wireup_store)  
    
    dist.barrier(device_ids=[world_rank])
def cleanup():
    dist.destroy_process_group()