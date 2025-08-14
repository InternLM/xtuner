# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys

from loguru import logger


_LOGGER = None


def log_format(debug: bool = False, rank: int | None = None):
    if rank is None:
        prefix = "[XTuner]"
    else:
        prefix = f"[XTuner][RANK {rank}]"
    formatter = f"{prefix}[{{time:YYYY-MM-DD HH:mm:ss}}][<level>{{level}}</level>]"

    if debug:
        formatter += "[<cyan>{name}</cyan>:"
        formatter += "<cyan>{function}</cyan>:"
        formatter += "<cyan>{line}</cyan>]"

    formatter += " <level>{message}</level>"
    return formatter


def get_logger(level="INFO"):
    global _LOGGER
    if _LOGGER is None:
        # Remove the original logger in Python to prevent duplicate printing.
        log_level = os.environ.get("XTUNER_LOG_LEVEL", level).upper()
        logger.remove()
        logger.add(sys.stderr, level=log_level, format=log_format(debug=log_level == "DEBUG"))
        _LOGGER = logger
    return _LOGGER
