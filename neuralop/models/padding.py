from torch.nn import functional as F
from torch import nn

class DomainPadding(nn.Module):
    """Applies domain padding scaled automatically to the input's resolution

    Parameters
    ----------
    domain_padding : float
        typically, between zero and one, percentage of padding to use
    padding_mode : {'symmetric', 'one-sided'}, optional
        whether to pad on both sides, by default 'one-sided'

    Notes
    -----
    This class works for any input resolution, as long as it is in the form
    `(batch-size, channels, d1, ...., dN)`
    """
    def __init__(self, domain_padding, padding_mode='one-sided'):
        super().__init__()
        self.domain_padding = domain_padding
        self.padding_mode = padding_mode.lower()
        
        # dict(f'{resolution}'=padding) such that padded = F.pad(x, indices)
        self._padding = dict()
        
        # dict(f'{resolution}'=indices_to_unpad) such that unpadded = x[indices]
        self._unpad_indices = dict()

    def forward(self, x):
        """forward pass: pad the input"""
        self.pad(x)
    
    def pad(self, x):
        """Take an input and pad it by the desired fraction
        
        The amount of padding will be automatically scaled with the resolution
        """
        resolution = x.shape[2:]

        if isinstance(self.domain_padding, (float, int)):
            self.domain_padding = [float(self.domain_padding)]*len(resolution)

        try:
            padding = self._padding[f'{resolution}']
            return F.pad(x, padding, mode='constant')

        except KeyError:
            padding = [int(round(p*r)) for (p, r) in zip(self.domain_padding, resolution)]
            
            print(f'Padding inputs of {resolution=} with {padding=}, {self.padding_mode}')

            if self.padding_mode == 'symmetric':
                # Pad both sides
                unpad_indices = (Ellipsis, ) + tuple([slice(p, -p, None) for p in padding])
                padding = [i for p in padding for i in (p, p)]

            elif self.padding_mode == 'one-sided':
                # One-side padding
                unpad_indices = (Ellipsis, ) + tuple([slice(None, -p, None) for p in padding])
                padding = [i for p in padding for i in (0, p)]
            else:
                raise ValueError(f'Got {self.padding_mode=}')
            
            self._padding[f'{resolution}'] = padding

            padded = F.pad(x, padding, mode='constant')
            self._unpad_indices[f'{padded.shape[2:]}'] = unpad_indices
            return padded

    def unpad(self, x):
        """Remove the padding from padding inputs
        """
        unpad_indices = self._unpad_indices[f'{x.shape[2:]}']

        return x[unpad_indices]
