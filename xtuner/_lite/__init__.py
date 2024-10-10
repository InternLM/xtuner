from loguru import logger

from .auto import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from .device import get_device, get_torch_device_module

# Remove the original logger in Python to prevent duplicate printing.
logger.remove()

_LOGGER = logger


def get_logger():
    return _LOGGER


__all__ = [
    'AutoConfig', 'AutoModelForCausalLM', 'AutoTokenizer', 'get_device',
    'get_torch_device_module'
]
