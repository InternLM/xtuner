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
@pytest.mark.parametrize("compile", [None, False])
def test_compile_model(model_cfg, compile):
    with LogCapture(logger) as cap:
        with torch.device("meta"):
            model = model_cfg(compile_cfg=compile).build()
        out = cap.get_output()

    matched = re.findall(r"\s([0-9a-zA-Z_\.]+) with options: (.*)", out)

    if compile == {} or compile is False:
        assert not matched, f"Expected no compile cfg in log, but got: {out}"
        assert model.compile_cfg == {}, f"Expected empty compile cfg, but got: {model.compile_cfg}"
        return

    if not matched:
        raise RuntimeError(f"Cannot find compile cfg in log: {out}")

    matched_compile_cfg = {}

    for key, val in matched:
        matched_compile_cfg[key] = literal_eval(val)


    compile_cfg = model.compile_cfg
    assert matched_compile_cfg == compile_cfg


def test_compile_model_exception():
    # Test nonexistent function
    with pytest.raises(Exception):
        with torch.device("meta"):
            Qwen3Dense8BConfig(compile_cfg={"a.b.c": {}}).build()

    # TOP level function without `@maybe_compile` cannot be defined in `compile_cfg`
    with pytest.raises(Exception):
        with torch.device("meta"):
            Qwen3Dense8BConfig(compile_cfg={"xtuner.v1.loss.utils.sp_split": {}}).build()

    with pytest.raises(Exception):
        with torch.device("meta"):
            Qwen3Dense8BConfig(compile_cfg={"xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEBlock.fuck": {}}).build()
