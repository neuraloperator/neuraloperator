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
