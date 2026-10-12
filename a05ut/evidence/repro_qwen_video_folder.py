"""用单测中的真实视频帧文件夹复现 Qwen 视频 tokenize 失败。"""

import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig


sample = json.loads(Path("tests/resource/mllm_sft_video_example_data_new.jsonl").read_text().splitlines()[7])
model_path = os.environ["QWEN3_5_MOE_PATH"]
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenize_fn = Qwen3VLTokenizeFnConfig(
    processor_path=model_path,
    chat_template="qwen3.5-vl",
    rand_video_max_frames=24,
    add_vision_id=False,
).build(tokenizer)
tokenize_fn.video_processor.fps = 3

result = tokenize_fn(sample, media_root=os.environ["VIDEO_ROOT"])
print(f"tokenized {len(result['input_ids'])} tokens")
