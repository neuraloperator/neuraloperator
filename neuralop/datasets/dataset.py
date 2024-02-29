from abc import ABCMeta

import torch
from torch.utils.data import DataLoader, Dataset

class PDEDataset(ABCMeta):
    def __init__(self):
        """PDEDataset is the base Dataset class for our library.
            Datasets contain input-output pairs a(x), u(x) and may also
            contain additional information, e.g. function parameters,
            input geometry or output query points.
        All datasets are required to expose the following attributes after init:

        train_db: torch.utils.data.Dataset of training examples
        test_db:  ""                       of test examples
        train_loader: torch.utils.data.DataLoader for single-node, single-GPU training
        test_loader: torch.utils.data.DataLoader for single-node, single-GPU training
        data_processor: neuralop.datasets.DataProcessor 
        """

