from ast import literal_eval
import re

import pytest
import torch

from xtuner._testing.utils import LogCapture
from xtuner.v1.model import (
    DeepSeekV3Config,
    GptOss21BA3P6Config,
    InternVL3P5Dense1BConfig,
    Qwen3Dense8BConfig,
    Qwen3MoE30BA3Config,
    Qwen3VLMoE30BA3Config,
)
from xtuner.v1.model.moe.moe import MOE_EP_COMPILE_CFG, MOE_NON_EP_COMPILE_CFG, MoE
from xtuner.v1.model.moe.qwen3_5_text import (
    MOE_EP_COMPILE_CFG as QWEN35_MOE_EP_COMPILE_CFG,
    MOE_NON_EP_COMPILE_CFG as QWEN35_MOE_NON_EP_COMPILE_CFG,
    Qwen3_5_VLTextMoE,
    Qwen3_5_VLTextMoE35BA3BConfig,
)
from xtuner.v1.module.dispatcher.base import (
    NaiveCombineResult,
    NaiveDispatchResult,
    NaivePreCombineResult,
    NaivePreDispatchResult,
)
from xtuner.v1.utils import get_logger


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


@pytest.mark.parametrize(
    "ep_size,expert_tp_size,expected_compile_cfg",
    [
        (1, 1, MOE_NON_EP_COMPILE_CFG),
        (2, 1, MOE_EP_COMPILE_CFG),
        (1, 2, MOE_EP_COMPILE_CFG),
        (2, 2, MOE_EP_COMPILE_CFG),
    ],
)
def test_moe_compile_cfg_treats_expert_tp_like_ep(ep_size, expert_tp_size, expected_compile_cfg):
    model = object.__new__(MoE)
    model.config = Qwen3MoE30BA3Config(ep_size=ep_size, expert_tp_size=expert_tp_size)
    assert model.default_compile_cfg == expected_compile_cfg


@pytest.mark.parametrize(
    "ep_size,expert_tp_size,expected_compile_cfg",
    [
        (1, 1, QWEN35_MOE_NON_EP_COMPILE_CFG),
        (2, 1, QWEN35_MOE_EP_COMPILE_CFG),
        (1, 2, QWEN35_MOE_EP_COMPILE_CFG),
        (2, 2, QWEN35_MOE_EP_COMPILE_CFG),
    ],
)
def test_qwen35_moe_compile_cfg_treats_expert_tp_like_ep(ep_size, expert_tp_size, expected_compile_cfg):
    model = object.__new__(Qwen3_5_VLTextMoE)
    model.config = Qwen3_5_VLTextMoE35BA3BConfig(ep_size=ep_size, expert_tp_size=expert_tp_size)
    assert model.default_compile_cfg == expected_compile_cfg


def test_naive_dispatcher_compile_result_typeddicts_have_no_optional_keys():
    # 中文注释：non-EP 默认会 compile MoEDecoderLayer.forward，Dynamo 不支持 optional-key TypedDict。
    assert NaivePreDispatchResult.__optional_keys__ == frozenset()
    assert NaiveDispatchResult.__optional_keys__ == frozenset()
    assert NaivePreCombineResult.__optional_keys__ == frozenset()
    assert NaiveCombineResult.__optional_keys__ == frozenset()
