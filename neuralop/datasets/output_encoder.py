from abc import ABC, abstractmethod
import torch

class OutputEncoder(torch.nn.Module):
    """OutputEncoder: converts the output of a model
        into a form usable by some cost function.
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
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

class UnitGaussianNormalizer(torch.nn.Module):
    """
    UnitGaussianNormalizer normalizes data to fit a 
    Gaussian distribution with unit variance. 
    """
    def __init__(self, data, dim=None, eps=1e-6):
        super().__init__()

        if isinstance(dim, int):
            dim = [dim]

        if isinstance(data, torch.Tensor):
            #Asumes batch dimension is first
            if dim is not None:
                if 0 not in dim:
                    dim.append(0)

            mean = torch.mean(data, dim, keepdim=True).squeeze(0)
            std = torch.std(data, dim, keepdim=True).squeeze(0)

        elif isinstance(data, list):
            total_n = self.get_total_elements(data[0], dim)
            mean = torch.mean(data[0], dim=dim, keepdim=True)
            squared_mean = torch.mean(data[0]**2, dim=dim, keepdim=True)

            for j in range(1, len(data)):
                current_n = self.get_total_elements(data[j], dim)

                mean = (1.0/(total_n + current_n))*(total_n*mean + torch.sum(data[j], dim=dim, keepdim=True))
                squared_mean = (1.0/(total_n + current_n))*(total_n*squared_mean + torch.sum(data[j]**2, dim=dim, keepdim=True))

                total_n += current_n
            
            std = torch.sqrt(squared_mean - mean**2)
            
        else:
            raise ValueError

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.register_buffer('eps', torch.tensor([eps]))
    
    def encode(self, x):
        x = x - self.mean
        x = x / (self.std + self.eps)

        return x
    
    def decode(self, x):
        x = x *(self.std + self.eps)
        x = x + self.mean

        return x

    def get_total_elements(sef, x, dim):
        n = 1
        if dim is not None:
            for d in dim:
                n *= x.shape[d]
        else:
            for j in range(len(x.shape)):
                n *= x.shape[j]
        
        return n
    
    def forward(self, x):
        return self.encode(x)
    
    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        self.eps = self.eps.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        self.eps = self.eps.cpu()
        return self
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.eps = self.eps.to(device)
        return self


class MultipleFieldOutputEncoder(OutputEncoder):
    """When a model has multiple output fields, 
        apply a different output encoder to each field. 
    
    Parameters
    -----------

    encoder_dict: dict
        dictionary of output encoders
    input_mappings: dict[tuple(Slice)]
        indices of an output tensor x to use for
        each field, such that x[mappings[field]]
        returns the correct slice of x.
    return_mappings: dict[tuple(Slice)]
        same as above. if only certain indices
        of encoder output are important, this indexes those.
    """
    def __init__(self, encoder_dict, input_mappings, return_mappings=None):
        self.encoders = encoder_dict
        self.output_fields = encoder_dict.keys()
        self.input_mappings = input_mappings
        self.return_mappings = return_mappings

        assert encoder_dict.keys() == input_mappings.keys()
        if self.return_mappings:
            assert encoder_dict.keys() == return_mappings.keys()

    def encode(self, x):
        """
        Parameters
        ----------
        x : Torch.tensor
            model output, indexed according to self.mappings
        """
        out = torch.zeros_like(x)
        
        for field,indices in self.input_mappings.items():
            encoded = self.encoders[field].encode(x[indices])
            if self.return_mappings:
                encoded = encoded[self.return_mappings[field]]
            out[indices] = encoded

        return out

    def decode(self, x):
        """
        Parameters
        ----------
        x : Torch.tensor
            model output, indexed according to self.mappings
        """
        out = torch.zeros_like(x)
        
        for field,indices in self.input_mappings.items():
            decoded = self.encoders[field].decode(x[indices])
            if self.return_mappings:
                decoded = decoded[self.return_mappings[field]]
            out[indices] = decoded
            
        return out
    
    def cpu(self):
        self.encoders = {k:v.cpu() for k,v in self.encoders.items()}

    def cuda(self):
        self.encoders = {k:v.cuda() for k,v in self.encoders.items()}

    def to(self, device):
        self.encoders = {k:v.to(device) for k,v in self.encoders.items()}