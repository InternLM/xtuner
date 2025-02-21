# Copyright (c) OpenMMLab. All rights reserved.
import os
import subprocess
import sys

from loguru import logger

from .device import get_device, get_torch_device_module

_LOGGER = None


def log_format(debug=False):
    formatter = "[XTuner][{time:YYYY-MM-DD HH:mm:ss}][<level>{level}</level>]"

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
        logger.remove()
        logger.add(sys.stderr, level=level, format=log_format(debug=level == "DEBUG"))
        _LOGGER = logger
    return _LOGGER


def get_repo_git_info(repo_path):
    original_directory = os.getcwd()
    os.chdir(repo_path)

    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.STDOUT
            )
            .strip()
            .decode("utf-8")
        )

        commit_id = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
            )
            .strip()
            .decode("utf-8")
        )

        remote_url = (
            subprocess.check_output(
                ["git", "remote", "get-url", "origin"], stderr=subprocess.STDOUT
            )
            .strip()
            .decode("utf-8")
        )

        return branch, commit_id, remote_url
    except subprocess.CalledProcessError:
        return None, None, None
    finally:
        os.chdir(original_directory)


__all__ = [
    "AutoConfig",
    "AutoModelForCausalLM",
    "AutoTokenizer",
    "get_device",
    "get_torch_device_module",
]
