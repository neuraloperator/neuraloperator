import torch.nn as nn

from .fno import FNO

class NestedFNO(nn.Module):
    """N level Nested FNO
    
    Parameters:
    ----------
    n_level : int tuple
        number of levels in the NestedFNO
    n_modes : list of tuple
        list of modes for each FNO model in NestedFNO
        ``len(n_modes)`` should euqal to n_level
    hidden_channels : list
        list of width (i.e. number of channels) of each FNO in NestedFNO 
    in_channels : int, optional
        number of (static) input channels, by default 3
    out_channels : int, optional
        number of output channels, by default 1
    lifting_channels : int, optional
        number of hidden channels of the lifting block of each FNO, by default 256
    projection_channels : int, optional
        number of hidden channels of the projection block of each FNO, by default 256
    """
    
    def __init__(
        self,
        n_level,
        n_modes,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        **kwargs
    ):
        super().__init__()
        self.n_level = n_level
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        # each coarser model's output is fed into finer model in Nested FNO, starting form the second model
        self.in_channels = [in_channels] + [in_channels + 1] * (self.n_level-1) 
        self.out_channels = [out_channels] * self.n_level
        self.lifting_channels = [lifting_channels] * self.n_level
        self.projection_channels = [projection_channels] * self.n_level

        assert self.n_level == len(self.n_modes), "n_modes does not match n_level"

        self.fnos = nn.ModuleList()
        for i in range(self.n_level):
            self.fnos.append(
                        FNO(n_modes=self.n_modes[i], 
                            hidden_channels=self.hidden_channels[i], 
                            in_channels=self.in_channels[i],
                            out_channels=self.out_channels[i],
                            lifting_channels=self.lifting_channels[i],
                            projection_channels=self.projection_channels[i],
                            use_mlp=False,
                            fno_skip='linear')
            )
            
    def forward(self, x):
        """
        x should be a list of input for each model
        Nested FNO will return a list of output form each model
        """
        
        y = [self.fnos[0](x[0])]
        
        for i in range(1,self.n_level):
            x[i] = torch.cat((x[i], y[i-1]), axis=1)
            y.append(self.fnos[i](x[i]))
            
        return y