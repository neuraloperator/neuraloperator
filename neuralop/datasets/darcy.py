
import torch
from .pde_dataset import BasePDEDataset, partialclass

    
'''def train_loader(self, 
                    num_workers: int=None, 
                    pin_memory: bool=True,
                    persistent_workers: bool=False) -> DataLoader:
    
    return DataLoader(dataset=self.train_db,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        persistent_workers=persistent_workers,
                        )

def test_loaders(self, 
                    num_workers: int=None, 
                    pin_memory: bool=True,
                    persistent_workers: bool=False) -> Dict[DataLoader]:
    test_loaders = {}
    for (res, batch_size) in zip(self.test_resolutions, self.test_batch_sizes):
        loader = DataLoader(dataset=self.test_dbs[res],
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            persistent_workers=persistent_workers,
                            )
        test_loaders[res] = loader
return test_loaders        '''

# Load small darcy flow as a partial class of DarcyFlowDataset
#SmallDarcyFlowDataset = partialclass(DarcyFlowDataset, train_resolution=16)