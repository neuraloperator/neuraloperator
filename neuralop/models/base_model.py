import inspect
import warnings

import torch
from torch import nn

def make_serializable(kwargs):
    for key, value in kwargs.items():
        if isinstance(value, BaseModel):
            kwargs[key] = {'BaseModel': value.state_dict()["_metadata"]}
        elif isinstance(value, dict):
            kwargs[key] = make_serializable(value)
        
        if isinstance(value, nn.Module):
            print(f"Dangerous value: {value}")
            warnings.warn("Warning: attempting to initialize a BaseModel with non-serializable kwargs. "
                          "In general, we recommend removing all nn.Modules from your model's init args."
                          )
    return kwargs

#  recursively loop through all torch.loaded metadata to turn deserialized init kwargs back into nested BaseModels
def deserialize_sub_models(kwargs):
    for key, value in kwargs.items():
        if isinstance(value, dict):
            base_model_init_kwargs = value.pop('BaseModel', None)
            if base_model_init_kwargs is not None:
                init_args = base_model_init_kwargs.pop('_args', [])
                base_model_cls = BaseModel._models[base_model_init_kwargs['_name'].lower()]
                kwargs[key] = base_model_cls(*init_args, **base_model_init_kwargs)
            else:
                kwargs[key] = deserialize_sub_models(value)
    return kwargs

class BaseModel(torch.nn.Module):
    """Base class for all Models

    This class has two main purposes:
    * It monitors the creation of subclasses that are automatically registered 
      for users to use by name using the library's config system
    * When a new instance of this class is created, the init call is intercepted
      so we can store the parameters used to create the instance.
      This makes it possible to save trained models along with their init parameters,
      and therefore load saved modes easily.

    Notes
    -----
    Any BaseModel instance can be versioned using the _version class attribute. 
    This can be used as a sanity check when loading models from checkpoints to verify the 
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
        metadata = cls._validate_and_store_args(sig, args, kwargs)
        instance = super().__new__(cls)
        instance._metadata = metadata

        return instance

    @classmethod
    def _validate_and_store_args(cls, sig, args, kwargs):
        class_name = cls.__name__

        # ensure that if metadata contains another BaseModel object, we convert that in a way that is loadable with ``weights_only=False``
        for i, arg in enumerate(args):
            if isinstance(arg, BaseModel):
                args[i] = {'BaseModel': arg._metadata}
        kwargs = make_serializable(kwargs)

        metadata = {"_args": args, "_kwargs": kwargs, "_version": cls._version, "_name": class_name}

        verbose = kwargs.get('verbose', False)
        # Unexpected arguments: verify that given parameters are actually arguments of the model
        for key in kwargs:
            if key not in sig.parameters and verbose:
                warnings.warn(f"Given argument '{key}' that isn't in the signature of class {class_name}.")

        # Fill in default arguments: check for model arguments not specified in the configuration
        for key, param in sig.parameters.items():
            if param.default is not inspect._empty and key not in kwargs:
                kwargs[key] = param.default
                if verbose:
                    print(f"Keyword argument {key} not specified for class {class_name},  using default={param.default}.")

        return metadata

    def state_dict(self, destination: dict=None, prefix: str='', keep_vars: bool=False):
        """
        state_dict subclasses nn.Module.state_dict() and adds a metadata field
        to track the model version and ensure only compatible saves are loaded.

        Parameters
        ----------
        destination : dict, optional
            If provided, the state of module will
            be updated into the dict and the same object is returned.
            Otherwise, an OrderedDict will be created and returned, by default None
        prefix : str, optional
            a prefix added to parameter and buffer
            names to compose the keys in state_dict, by default ``''``
        keep_vars (bool, optional): by default the torch.Tensors
            returned in the state dict are detached from autograd. 
            If True, detaching will not be performed, by default False

        """
        torch_state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return {
            "state": torch_state_dict,
            "_metadata": self._metadata
        }

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """load_state_dict subclasses nn.Module.load_state_dict() and adds a metadata field
        to track the model version and ensure only compatible saves are loaded.

        Parameters
        ----------
        state_dict : dict
            state dictionary generated by ``nn.Module.state_dict()``
            .. note ::
                In a ``BaseModel`` this is keyed
                ``{'state': super().state_dict(), '_metadata': self.metadata}``

        strict : bool, optional
            whether to strictly enforce that the keys in ``state_dict``
            match the keys returned by this module's, by default True.
        assign : bool, optional
            whether to assign items in the state dict to their corresponding keys
            in the module instead of copying them inplace into the module's current
            parameters and buffers. When False, the properties of the tensors in the
            current module are preserved while when True, the properties of the Tensors
            in the state dict are preserved, by default False

        Returns
        -------
        _type_
            _description_
        """
        metadata = state_dict.pop('_metadata', None)

        if metadata is not None:
            saved_version = metadata.get('_version', None)
            if saved_version is None:
                warnings.warn(f"Saved instance of {self.__class__} has no stored version attribute.")
            if saved_version != self._version:
                warnings.warn(f"Attempting to load a {self.__class__} of version {saved_version},"
                              f"But current version of {self.__class__} is {saved_version}")
            # remove state dict metadata at the end to ensure proper loading with PyTorch module
        super().load_state_dict(state_dict['state'], strict=strict, assign=assign)
        self._metadata = metadata
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, map_location=None, strict=True, assign=False):
        # Load the checkpoint safely: change weights_only to False if you want to load the full checkpoint
        state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=True)

        metadata = state_dict.get('_metadata', dict())
        version = metadata.get('_version', None)

        if version is not None and hasattr(cls, '_version') and version != cls._version:
            warnings.warn(f'Checkpoint saved for version {version} of class {cls.__name__} but current code is version {cls._version}')

        metadata = deserialize_sub_models(metadata)

        init_args = metadata.get('_args', list())
        init_kwargs = metadata.get('_kwargs', dict())
        instance = cls(*init_args, **init_kwargs)

        instance.load_state_dict(state_dict, strict=strict, assign=assign)
        instance._metadata = metadata

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
    arch = config.model['model_arch'].lower()
    model_config = config.model

    # Set the number of input channels depending on channels in data + mg patching
    data_channels = model_config.pop("data_channels")
    try:
        patching_levels = config["patching"]["levels"]
    except KeyError:
        patching_levels = 0
    if patching_levels:
        data_channels *= patching_levels + 1
    model_config["in_channels"] = data_channels

    # Dispatch model creation
    try:
        return BaseModel._models[arch](**model_config)
    except KeyError:
        raise ValueError(f"Got config.arch={arch}, expected one of {available_models()}.")