# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import threading

import loguru
from loguru import logger
from mmengine.dist import get_rank


_LOGGER = None
_LOGGER_LOCK = threading.Lock()


def log_format(debug: bool = False, module: str | None = None, rank: int | None = None):
    if rank is None:
        prefix = "[XTuner]"
    else:
        prefix = f"[XTuner][RANK {rank}]"

    formatter = f"{prefix}[{{time:YYYY-MM-DD HH:mm:ss}}][<level>{{level}}</level>]"

    if module is not None:
        formatter += f"[{module}]"

    if debug:
        formatter += "[<cyan>{name}</cyan>:"
        formatter += "<cyan>{function}</cyan>:"
        formatter += "<cyan>{line}</cyan>]"

    formatter += " <level>{message}</level>"
    return formatter


def get_logger(level="INFO", log_dir=None, tag=None):
    global _LOGGER
    if _LOGGER is None:
        with _LOGGER_LOCK:
            if _LOGGER is None:
                # Remove the original logger in Python to prevent duplicate printing.
                log_level = os.environ.get("XTUNER_LOG_LEVEL", level).upper()
                logger.remove()
                logger.add(sys.stderr, level=log_level, format=log_format(debug=log_level == "DEBUG"), enqueue=True)
                _LOGGER = logger

    # note: 部分推理后端会清除logger的handler，导致日志输出至文件有异常，所以这里重新add file handler
    if log_dir is not None:
        log_level = os.environ.get("XTUNER_LOG_LEVEL", level).upper()
        add_log_dir = True
        for handler in _LOGGER._core.handlers.values():
            if isinstance(handler._sink, loguru._file_sink.FileSink):
                add_log_dir = False
                break
        # 保证只打印一次
        if add_log_dir:
            _LOGGER.add(
                log_dir / f"rank_{get_rank()}.log",
                format=log_format(rank=get_rank(), module=tag if tag else None),
                level=log_level,
                enqueue=True,
            )
    return _LOGGER
