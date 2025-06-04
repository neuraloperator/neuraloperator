from ..transforms.data_processors import DefaultDataProcessor
try:
    from einops import rearrange
except ModuleNotFoundError:
    print("Trying to import data processors for TheWell without required dependency\
          ``einops``. Run ``pip install einops`` and try again.")
    raise ModuleNotFoundError

import torch

class TheWellDataProcessor(DefaultDataProcessor):
    """
    TheWellDataProcessor converts data from ``neuralop.data.datasets.TheWellDataset``
    into the form expected by our models. 

    Parameters
    ----------
    data_normalizer : nn.Module
        data transform to normalize/unnormalize time-varying spatial variables in x and y
        channel-wise. The problems in The Well are trajectories in time,
        so x and y are points from the same trajectory and thus share statistics.
    const_normalizer : nn.Module, optional
        data transform to normalize/unnormalize constant input variables in x
        channel-wise. If they exist, these constant fields are concatenated onto x
        after normalizing and flattening. 
    n_steps_input : int
        number of timesteps to be used as input to the model
    n_steps_output : int
        number of timesteps to be used as output to the model
    time_as_channels : bool, optional
        _description_, by default True
    """
        
    def __init__(self, data_normalizer, 
                 const_normalizer=None, 
                 n_steps_input: int=1, 
                 n_steps_output: int=1, 
                 time_as_channels: bool=True):
        
        super().__init__()
        self.data_normalizer = data_normalizer
        self.const_normalizer = const_normalizer
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.time_as_channels = time_as_channels

        if self.time_as_channels:
            assert self.n_steps_output == 1, "if attempting to predict multiple timesteps of output, use a spatiotemporal model instead of flattening."

    def to(self, device):
        self.device = device
        if self.data_normalizer is not None:
            self.data_normalizer = self.data_normalizer.to(self.device)
        if self.const_normalizer is not None:
            self.const_normalizer = self.const_normalizer.to(self.device)
        return self

    def preprocess(self, data_dict, step=None):
        """
        Code adapted from the_well.data.data_formatter.DefaultChannelsFirstFormatter
        """
        ## processing x
        # in next-step mode, or if x is not yet set
        if step is None:

            # by default, in The Well, x is stored in shape (b, time, d1, d2, ... dN, channels)
            x = data_dict["input_fields"].to(self.device)
            x = x.permute(0,-1, *list(range(1, x.ndim - 1)))
            data_dict["input_fields"] = x # store permuted version for later
        elif step == 0:
            x = data_dict["output_fields"][:, :self.n_steps_input, ...].to(self.device)

            # always move channels to the second dimension
            x = x.permute(0,-1, *list(range(1, x.ndim - 1)))
            data_dict["input_fields"] = x # store permuted version for later
            in_channels = x.shape[1]
        else:
            # otherwise input fields are set in self.postprocess (see below)
            x = data_dict["input_fields"].to(self.device)
        if self.data_normalizer is not None:
            x = self.data_normalizer.transform(x)

        # Normalization is performed channel-wise, so we flatten time along the channel dim once data has been normalized
        if self.time_as_channels:
            x = rearrange(x, "b c t ... -> b (t c) ...")
        
        # Retrieve, normalize and concat constant variable channels if they exist

        if "constant_fields" in data_dict:
            # TODO repeat along time 
            flat_constants = rearrange(data_dict["constant_fields"], "b ... c -> b c ...")
            constant_channels = flat_constants.shape[1]
            
            if self.const_normalizer is not None:
                flat_constants = self.const_normalizer.transform(flat_constants)
            
            # if x stays spatiotemporal, repeat constants along time dim
            if not self.time_as_channels:
                flat_constants = flat_constants.unsqueeze(2).repeat(1, 1, x.shape[2], *[1]*(x.ndim - 3))
            
            # concatenate along channels after normalizing
            x = torch.cat(
                [
                    x,
                    flat_constants.to(self.device),
                ],
                dim=1,
            )
        
        ### Processing y
        # by default, in The Well, y is stored in shape (b, time, d1, d2, ... dN, channels)
        y = data_dict["output_fields"].to(self.device)
        # always move channels to the second dimension
        # shape (b, c, t, ...)
        y = y.permute(0,-1, *list(range(1, y.ndim - 1)))

        # if in autoregressive (AR) mode, skip default preprocessing, infer number of steps and roll y forward
        if step is not None:
            # the first n_steps_input steps are reserved for x.
            step += self.n_steps_input 

            # if we're past the end of y's timesteps, return an empty sample.
            if step >= y.shape[2]:
                return None
            else:
                # here y is shape (b, c, t, ...)
                # roll y forward one timestep
                y = y[:, :, step:step+self.n_steps_output, ...]
        
        # normalize y after selecting only the relevant timesteps
        if self.data_normalizer is not None:
            y = self.data_normalizer.transform(y)
        
        # In both modes, permute y's dimensions into the same order as x's
        if self.time_as_channels:
            y = rearrange(y, "b c t ... -> b (t c) ...")
        
        data_dict["x"] = x
        data_dict["y"] = y

        return data_dict

    def postprocess(self, output, data_dict, step=None):
        """
        Code adapted from the_well.data.data_formatter.DefaultChannelsFirstFormatter
        """
        
        y = data_dict["y"]

        # Unnormalize: outputs/ground truth/(sometimes inputs)
        if self.data_normalizer is not None:

            # in next-step mode, out and y are unnormalized unly during eval.
            if (step is None and not self.training) or step is not None:
                # before unnormalizing, reshape out + y to add a time dim
                # we can unsqueeze because we assume n_steps_output == 1.
                if self.time_as_channels:
                    y = y.unsqueeze(2)
                    output = output.unsqueeze(2)

                y = self.data_normalizer.inverse_transform(y)
                output = self.data_normalizer.inverse_transform(output)

                # after normalizing, squeeze time dim out again (note n_steps_output forced to 1)
                if self.time_as_channels:
                    y = y.squeeze(2)
                    output = output.squeeze(2)

        # if in AR mode, append output to x
        if step is not None:
            # only grab variable fields
            input_vars = data_dict["input_fields"].to(self.device)
            # on step=0, the channels are permuted to dim 1
            # concatenate along time dim (add to output)
            input_vars = torch.cat((input_vars, output.unsqueeze(2)), dim=2)
            # roll forward by one step
            input_vars = input_vars[:, :, -self.n_steps_input:, ...]
            data_dict["input_fields"] = input_vars

        data_dict["y"] = y
        return output, data_dict
