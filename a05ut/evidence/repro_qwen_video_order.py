"""Compare real numeric video frames with the historical Qwen sort key."""

import os
import re
from pathlib import Path

from xtuner.v1.datasets.mllm_tokenize_fn.qwen3_vl_utils import sort_frames


for folder in ("tennis_frames_4fps", "tennis_frames_2fps"):
    listed = os.listdir(Path(os.environ["VIDEO_ROOT"]) / folder)
    historical_keys = [re.search(r"_(\d+).jpg$", name) for name in listed]
    ordered = sort_frames(listed)
    print(f"{folder}: {len(listed)} frames, old_keys_all_missing={all(key is None for key in historical_keys)}")
    print(f"filesystem first 8: {listed[:8]}")
    print(f"temporal first 8:   {ordered[:8]}")
    assert listed != ordered
