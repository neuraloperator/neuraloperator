from torch import nn


class BaseSpectralConv(nn.Module):
    def __init__(self, device=None, dtype=None):
        """Base Class for Spectral Convolutions
        
        Use it when you want to build your own FNO-type Neural Operators
        """
        super().__init__()

        self.dtype = dtype
        self.device = device

    def transform(self, x):
        """Transforms an input x for a skip connection, by default just an identity map 

        If your function transforms the input then you should also implement this transform method 
        so the skip connection can also work. 

        Typical usecases are:
        * Your upsample or downsample the input in the Spectral conv: the skip connection has to be similarly scaled. 
           THis allows you to deal with it however you want (e.g. avoid aliasing)
        * You perform a change of basis in your Spectral Conv, again, this needs to be applied to the skip connection too.
        """
        return x
