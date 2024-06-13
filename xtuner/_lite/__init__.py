from loguru import logger

from .auto import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Remove the original logger in Python to prevent duplicate printing.
logger.remove()

_LOGGER = logger


def get_logger():
    return _LOGGER


__all__ = ['AutoConfig', 'AutoModelForCausalLM', 'AutoTokenizer']
