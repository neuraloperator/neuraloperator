from torch.utils.data.dataset import Dataset


class TensorDataset(Dataset):
    def __init__(self, x, y, transform_x=None, transform_y=None):
        assert (x.size(0) == y.size(0)), "Size mismatch between tensors"
        self.x = x
        self.y = y
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        
        if self.transform_x is not None:
            x = self.transform_x(x)

        if self.transform_y is not None:
            x = self.transform_y(x)

        return {'x': x, 'y':y}

    def __len__(self):
        return self.x.size(0)

class GeneralTensorDataset(Dataset):
    def __init__(self, sets, transforms):
        assert len(sets) == len(transforms), "Size mismatch between number of tensors and transforms"
        self.n = len(sets)
        if self.n > 1:
            for j in range(1,self.n):
                assert sets[j].size(0) == sets[0].size(0), "Size mismatch between tensors"
        
        self.sets = sets
        self.transforms = transforms

    def __getitem__(self, index):
        if self.n > 1:
            items = []
            for j in range(self.n):
                items.append(self.sets[j][index])
                if self.transforms[j] is not None:
                    items[j] = self.transforms[j](items[j])
        else:
            items = self.sets[0][index]
            if self.transforms[0] is not None:
                    items = self.transforms[0](items)
        
        return items

    def __len__(self):
        return self.sets[0].size(0)