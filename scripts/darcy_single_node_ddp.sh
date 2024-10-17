#!/bin/bash

# Replace NUM_GPUS with the number of GPUs on your node
torchrun --standalone --nproc_per_node NUM_GPUS train_darcy.py \
    --distributed.use_distributed True