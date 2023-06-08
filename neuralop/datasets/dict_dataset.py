from torch.utils.data import Dataset

class DictDataset(Dataset):
    def __init__(self,
                 data_list: list,
                 constant: dict = None,
                 ):
        
        self.data_list = data_list
        self.constant = constant

    def __getitem__(self, index):
        return_dict = self.data_list[index]

        if self.constant is not None:
            return_dict.update(self.constant)
        
        return return_dict
        
    def __len__(self):
        return len(self.data_list)
