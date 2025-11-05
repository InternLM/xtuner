# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import threading

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
    if log_dir:
        log_level = os.environ.get("XTUNER_LOG_LEVEL", level).upper()
        add_main_log_dir = True
        add_infer_log_dir = True
        for handler in _LOGGER._core.handlers.values():
            if handler._name == repr(str(log_dir / f"rank_{get_rank()}.log")):
                add_main_log_dir = False
            if handler._name == repr(str(log_dir / "infer_engine_error.log")):
                add_infer_log_dir = False

        # 保证只打印一次
        if add_main_log_dir:
            logger.add(
                log_dir / f"rank_{get_rank()}.log",
                filter=lambda record: record["extra"].get("tag") != "InferEngine",
                format=log_format(rank=get_rank(), module=tag if tag else None),
                level=log_level,
                enqueue=True,
            )
        if add_infer_log_dir:
            logger.add(
                log_dir / "infer_engine_error.log",
                # 关键：过滤器只允许 tag 为 'infer engine' 的日志通过
                filter=lambda record: record["extra"].get("tag") == "InferEngine",
                format=log_format(module="infer_engine"),
                level=log_level,
                enqueue=True,
            )
    _LOGGER = logger

    return _LOGGER
