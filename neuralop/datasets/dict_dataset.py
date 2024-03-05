from typing import List

from torch.utils.data import Dataset


class DictDataset(Dataset):
    """DictDataset is a basic dataset form that stores each batch
    as a dictionary of tensors or other data structures


    """

    def __init__(
        self,
        data_list: List[dict],
        constant: dict = None,
    ):
        """

        Parameters
        ----------
        data_list : List[dict]
            list of individual batch dictionaries
        constant : dict, optional
            if each data batch shares some constant valued key/val pairs,
            they can be stored in constant for simplicity
        """

        self.data_list = data_list
        self.constant = constant

    def __getitem__(self, index):
        return_dict = self.data_list[index]

        if self.constant is not None:
            return_dict.update(self.constant)

        return return_dict

    def __len__(self):
        return len(self.data_list)
