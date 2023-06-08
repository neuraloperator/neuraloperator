import torch

class UnitGaussianNormalizer(torch.nn.Module):
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
