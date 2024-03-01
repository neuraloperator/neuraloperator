from abc import ABCMeta, abstractmethod

from torch.utils.data import DataLoader
from .data_transforms import DataProcessor

class PDEDataset(ABCMeta):
    def __init__(self):
        """PDEDataset is the base Dataset class for our library.
            Datasets contain input-output pairs a(x), u(x) and may also
            contain additional information, e.g. function parameters,
            input geometry or output query points.

            datasets may implement a download flag at init, which provides
            access to a number of premade datasets for sample problems provided
            in our Zenodo archive. 

        All datasets are required to expose the following attributes after init:

        train_db: torch.utils.data.Dataset of training examples
        test_db:  ""                       of test examples
        train_loader: torch.utils.data.DataLoader for single-node, single-GPU training
        test_loaders: torch.utils.data.DataLoader (s) for single-node, single-GPU training
        data_processor: neuralop.datasets.DataProcessor to process data examples
            optional, default is None
        """
        pass
    
    @abstractmethod
    def train_loader(self) -> DataLoader:
        pass
    
    @abstractmethod
    def test_loaders(self) -> DataLoader:
        pass
    
    def data_processor(self) -> DataProcessor:
        # Optional: provides a DataProcessor if overriden
        return None
    
def download_fn(pth: str):
    # print fname for now, later will be a real function
    return pth
