# Copyright (c) OpenMMLab. All rights reserved.
import os


def is_deterministic() -> bool:
    """Check if the running environment is deterministic."""
    return os.getenv("XTUNER_DETERMINISTIC") == "true"
