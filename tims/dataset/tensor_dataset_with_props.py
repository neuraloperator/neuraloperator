from torch.utils.data.dataset import Dataset


class TensorDatasetWithProps(Dataset):
    def __init__(self, x, y, props, transform_x=None, transform_y=None):
        assert x.size(0) == y.size(0), "Size mismatch between tensors"
        assert x.size(0) == len(props), "Size mismatch between tensors and properties dict"
        self.x = x
        self.y = y
        self.props = props
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        props = self.props[index]
        if self.transform_x is not None:
            x = self.transform_x(x)

        if self.transform_y is not None:
            y = self.transform_y(y)

        return {"x": x, "y": y , "props": props}    

    def __len__(self):
        return self.x.size(0)