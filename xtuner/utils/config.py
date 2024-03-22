import dataclasses
from typing import TypeVar, Union

import torch
from mmengine.config import Config
from mmengine.logging import print_log
from mmengine.utils import get_object_from_string

from xtuner.registry import BUILDER


def convert_dtype_cfg_to_obj(config: Union[dict, list[dict]]) -> None:
    """Convert dtype related config to python object.

    When MMEngine Runner is training, it will save the config file for
    resuming training.
    But in the saved config file, python objects of type torch.dtype are
    converted to strings like 'torch.float16'. In order to accommodate this,
    after loading the config, all dtype strings need to be converted into
    python objects.

    Args:
        config: A dict or list that potentially contains dtypes as strings.

    Returns:
        None. The input 'config' is modified in-place.
    """
    # If the config is a dictionary
    if isinstance(config, dict):
        for key, value in config.items():
            # Recursively call the function if the value is also a dict
            if isinstance(value, dict):
                convert_dtype_cfg_to_obj(value)

            # Replace the string with the corresponding dtype object
            # if it's a recognized dtype string
            elif value in ['torch.float16', 'torch.float32', 'torch.bfloat16']:
                config[key] = getattr(torch, value.split('.')[-1])

    # If the config is a list
    elif isinstance(config, list):
        for item in config:
            convert_dtype_cfg_to_obj(item)


def convert_dataclass_cfg_to_obj(config: Union[dict, list[dict]]) -> None:
    """Convert dataclass related config to python object.

    Huggingface's code uses dataclasses extensively.
    In order to use Huggingface's interfaces in the MMEngine config,
    we need to specifically handle these configurations.

    Note:
        Before executing this function, you must first run
        `convert_dtype_cfg_to_obj`, otherwise the dataclass config containing
        dtype cannot be properly converted !

    Args:
        config: A dictionary or list that potentially contains configurations
            as dataclasses.

    Returns:
        None. The input 'config' is modified in-place.
    """
    # If the config is a dictionary
    if isinstance(config, dict):
        for key, value in config.items():
            # Recursively call the function if the value is also a dict
            if isinstance(value, dict):
                convert_dataclass_cfg_to_obj(value)

                # Check if the type of value is a dataclass
                if 'type' in value and dataclasses.is_dataclass(value['type']):
                    builder = value.pop(
                        'type')  # remove 'type' from value and get its content

                    # Convert the builder to an object if it is a string
                    if isinstance(builder, str):
                        builder = get_object_from_string(builder)

                    # Build a new_value using the remaining items in value
                    new_value = builder(**value)
                    # replace the original value with new_value
                    config[key] = new_value
                    print_log(f'{key} convert to {builder}')

    # If the config is a list
    elif isinstance(config, list):
        for item in config:
            convert_dataclass_cfg_to_obj(item)


OBJ_T = TypeVar('OBJ_T')


def build_from_cfg_or_obj(cfg_or_obj: Union[dict, OBJ_T],
                          accept: OBJ_T) -> OBJ_T:
    """Build a python object from a config or return an existed object.

    Args:
        cfg_or_obj (dict, OBJ_T]):  an object of a type specified in
            `accept_obj_types`, or a dict.
        accept_obj (OBJ_T): the type of object that return without any
            modification.

    Returns:
        If 'cfg_or_obj' is an object of `accept_obj` , it is returned directly.
        If 'cfg_or_obj' is a dict, it is built into an object using
        `build_from_cfg()`.

    Raises:
        TypeError: If `cfg_or_obj` is not dict or `accept_obj`.
    """

    if isinstance(cfg_or_obj, accept):
        return cfg_or_obj

    elif isinstance(cfg_or_obj, (dict, Config)):
        convert_dtype_cfg_to_obj(cfg_or_obj)
        convert_dataclass_cfg_to_obj(cfg_or_obj)
        obj = BUILDER.build(cfg_or_obj)

        if not isinstance(obj, accept):
            raise TypeError(
                f'Expect an object of {accept}, but there is an object of '
                f'{type(obj)}.')
        return obj

    else:
        raise TypeError(f'cfg_or_obj must be a dict, or {accept}, but got '
                        f'{type(cfg_or_obj)}')
