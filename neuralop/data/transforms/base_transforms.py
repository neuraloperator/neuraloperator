from abc import abstractmethod
from typing import List

import torch

class Transform(torch.nn.Module):
    """
    Applies transforms or inverse transforms to 
    model inputs or outputs, respectively
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def inverse_transform(self):
        pass

    @abstractmethod
    def cuda(self):
        pass

    @abstractmethod
    def cpu(self):
        pass

    @abstractmethod
    def to(self, device):
        pass

class CompositeTransform(Transform):
    def __init__(self, transforms: List[Transform]):
        """Composite transform composes a list of
        Transforms into one Transform object.

        Transformations are not assumed to be commutative

        Parameters
        ----------
        transforms : List[Transform]
            list of transforms to be applied to data
            in order
        """
        super.__init__()
        self.transforms = transforms
    
    def transform(self, data_dict):
        for tform in self.transforms:
            data_dict = tform.transform(self.data_dict)
        return data_dict
    
    def inverse_transform(self, data_dict):
        for tform in self.transforms[::-1]:
            data_dict = tform.transform(self.data_dict)
        return data_dict

    def to(self, device):
        # all Transforms are required to implement .to()
        self.transforms = [t.to(device) for t in self.transforms if hasattr(t, 'to')]
        return self

class DictTransform(Transform):
    """When a model has multiple input and output fields,
        apply a different transform to each field,
        tries to apply the inverse_transform to each output

    Parameters
    -----------

    transform_dict: dict
        dictionary of output encoders
    input_mappings: dict[tuple(Slice)]
        indices of an output tensor x to use for
        each field, such that x[mappings[field]]
        returns the correct slice of x.
    return_mappings: dict[tuple(Slice)]
        same as above. if only certain indices
        of encoder output are important, this indexes those.
    """

    def __init__(self, transform_dict, input_mappings, return_mappings=None):
        self.transforms = transform_dict
        self.output_fields = transform_dict.keys()
        self.input_mappings = input_mappings
        self.return_mappings = return_mappings

        assert transform_dict.keys() == input_mappings.keys(), \
        f"Error: expected keys in transform_dict and input_mappings to match,\
             received {transform_dict.keys()=}\n{input_mappings.keys()=}"
        if self.return_mappings:
            assert transform_dict.keys() == return_mappings.keys(), \
        f"Error: expected keys in transform_dict and return_mappings to match,\
             received {transform_dict.keys()=}\n{return_mappings.keys()=}"

    def transform(self, tensor_dict):
        """
        Parameters
        ----------
        tensor_dict : Torch.tensor dict
            model output, indexed according to self.mappings
        """
        out = torch.zeros_like(tensor_dict)

        for field, indices in self.input_mappings.items():
            encoded = self.transforms[field].transform(tensor_dict[indices])
            if self.return_mappings:
                encoded = encoded[self.return_mappings[field]]
            out[indices] = encoded

        return out

    def inverse_transform(self, x):
        """
        Parameters
        ----------
        x : Torch.tensor
            model output, indexed according to self.mappings
        """
        out = torch.zeros_like(x)
        for field, indices in self.input_mappings.items():
            decoded = self.transforms[field].inverse_transform(x[indices])
            print(f"{decoded.shape=}")
            if self.return_mappings:
                decoded = decoded[self.return_mappings[field]]
            out[indices] = decoded

        return out

    def cpu(self):
        self.encoders = {k: v.cpu() for k, v in self.transforms.items()}

    def cuda(self):
        self.encoders = {k: v.cuda() for k, v in self.transforms.items()}

    def to(self, device):
        self.encoders = {k: v.to(device) for k, v in self.transforms.items()}
        return self
