from xtuner.v1.model import Qwen3Dense8BConfig, Qwen3MoE30BA3Config, Qwen3VLMoE30BA3Config, GptOss21BA3P6Config, DeepSeekV3Config, InternVL3P5Dense1BConfig, XTunerBaseModelConfig
import torch
from xtuner.v1.utils import get_logger
from xtuner._testing.utils import LogCapture
from ast import literal_eval
import pytest

import re


logger = get_logger()


@pytest.mark.parametrize("model_cfg", [
    Qwen3Dense8BConfig,
    Qwen3MoE30BA3Config,
    Qwen3VLMoE30BA3Config,
    GptOss21BA3P6Config,
    InternVL3P5Dense1BConfig,
    DeepSeekV3Config,
])
def test_compile_model(model_cfg):
    with LogCapture(logger) as cap:
        with torch.device("meta"):
            model = model_cfg().build()
        out = cap.get_output()


    matched = re.findall(r"\s([0-9a-zA-Z_\.]+) with options: (.*)", out)

    if not matched:
        raise RuntimeError(f"Cannot find compile cfg in log: {out}")

    matched_compile_cfg = {}

    for key, val in matched:
        matched_compile_cfg[key] = literal_eval(val)


    compile_cfg = model.compile_cfg
    assert matched_compile_cfg == compile_cfg
