import inspect
import torch
import warnings
from pathlib import Path

# Author: Jean Kossaifi

class BaseModel(torch.nn.Module):
    """Based class for all Models

    This class has two main functionalities:
    * It monitors the creation of subclass, that are automatically registered 
      for users to use by name using the library's config system
    * When a new instance of this class is created, the init call is intercepted
      so we can store the parameters used to create the instance.
      This makes it possible to save trained models along with their init parameters,
      and therefore load saved modes easily.

    Notes
    -----
    Model can be versioned using the _version class attribute. 
    This can be used for sanity check when loading models from checkpoints to verify the 
    model hasn't been updated since.
    """
    _models = dict()
    _version = '0.1.0'

    def __init_subclass__(cls, name=None, **kwargs):
        """When a subclass is created, register it in _models
        We look for an existing name attribute. 
        If not give, then we use the class' name.
        """
        super().__init_subclass__(**kwargs)
        if name is not None:
            BaseModel._models[name.lower()] = cls
            cls._name = name
        else:
            # warnings.warn(f'Creating a subclass of BaseModel {cls.__name__} with no name, initializing with {cls.__name__}.')
            BaseModel._models[cls.__name__.lower()] = cls
            cls._name = cls.__name__

    def __new__(cls, *args, **kwargs):
        """Verify arguments and save init kwargs for loading/saving

        We inspect the class' signature and check for unused parameters, or 
        parameters not passed. 

        We store all the args and kwargs given so we can duplicate the instance transparently.
        """
        sig = inspect.signature(cls)
        model_name = cls.__name__

        verbose = kwargs.get('verbose', False)
        # Verify that given parameters are actually arguments of the model
        for key in kwargs:
            if key not in sig.parameters:
                if verbose:
                    print(f"Given argument key={key} "
                        f"that is not in {model_name}'s signature.")

        # Check for model arguments not specified in the configuration
        for key, value in sig.parameters.items():
            if (value.default is not inspect._empty) and (key not in kwargs):
                if verbose:
                    print(
                        f"Keyword argument {key} not specified for model {model_name}, "
                        f"using default={value.default}."
                    )
                kwargs[key] = value.default

        if hasattr(cls, '_version'):
            kwargs['_version'] = cls._version
        kwargs['args'] = args
        kwargs['_name'] = cls._name
        instance = super().__new__(cls)
        instance._init_kwargs = kwargs

        return instance
    
    def save_checkpoint(self, save_folder, save_name):
        """Saves the model state and init param in the given folder under the given name
        """
        save_folder = Path(save_folder)

        state_dict_filepath = save_folder.joinpath(f'{save_name}_state_dict.pt').as_posix()
        torch.save(self.state_dict(), state_dict_filepath)
        metadata_filepath = save_folder.joinpath(f'{save_name}_metadata.pkl').as_posix()
        # Objects (e.g. GeLU) are not serializable by json - find a better solution in the future
        torch.save(self._init_kwargs, metadata_filepath)
        # with open(metadata_filepath, 'w') as f:
        #     json.dump(self._init_kwargs, f)

    def load_checkpoint(self, save_folder, save_name):
        save_folder = Path(save_folder)
        state_dict_filepath = save_folder.joinpath(f'{save_name}_state_dict.pt').as_posix()
        self.load_state_dict(torch.load(state_dict_filepath))
    
    @classmethod
    def from_checkpoint(cls, save_folder, save_name):
        save_folder = Path(save_folder)

        metadata_filepath = save_folder.joinpath(f'{save_name}_metadata.pkl').as_posix()
        init_kwargs = torch.load(metadata_filepath)
        # with open(metadata_filepath, 'r') as f:
        #     init_kwargs = json.load(f)
        
        version = init_kwargs.pop('_version')
        if hasattr(cls, '_version') and version != cls._version:
            print(version)
            warnings.warn(f'Checkpoing saved for version {version} of model {cls._name} but current code is version {cls._version}')
        
        if 'args' in init_kwargs:
            init_args = init_kwargs.pop('args')
        else:
            init_args = []
        instance = cls(*init_args, **init_kwargs)

        instance.load_checkpoint(save_folder, save_name)
        return instance


def available_models():
    """List the available neural operators"""
    return list(BaseModel._models.keys())


def get_model(config):
    """Returns an instantiated model for the given config

    * Reads the model to be used from config['arch']
    * Adjusts config["arch"]["data_channels"] accordingly if multi-grid patching is used

    Also prints warnings for safety, in case::
    * some given arguments aren't actually used by the model
    * some keyword arguments of the models aren't provided by the config

    Parameters
    ----------
    config : Bunch or dict-like
        configuration, must have
        arch = config['arch'] (string)
        and the corresponding config[arch] (a subdict with the kwargs of the model)

    Returns
    -------
    model : nn.Module
        the instanciated module
    """
    arch = config["arch"].lower()
    config_arch = config.get(arch)

    # Set the number of input channels depending on channels in data + mg patching
    data_channels = config_arch.pop("data_channels")
    try:
        patching_levels = config["patching"]["levels"]
    except KeyError:
        patching_levels = 0
    if patching_levels:
        data_channels *= patching_levels + 1
    config_arch["in_channels"] = data_channels

    # Dispatch model creation
    try:
        return BaseModel._models[arch](**config_arch)
    except KeyError:
        raise ValueError(f"Got config.arch={arch}, expected one of {available_models()}.")