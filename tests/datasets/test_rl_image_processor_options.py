"""CPU image-budget propagation, without starting an inference server."""

import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from transformers import AutoProcessor, AutoTokenizer
from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams
from xtuner.v1.datasets.rl_tokenize_fn.glm53_vl_tokenize_fn import RLGlm53VLTokenizeFnConfig
from xtuner.v1.datasets.rl_tokenize_fn.qwen3_vl_tokenize_fn import RLQwen3VLTokenizeFnConfig
from xtuner.v1.rl.distillation.rollout_teacher_manager import RolloutTeacherClient


def actor_payload(state, return_token_ids=True):
    # Execute the real payload method, avoiding imports of optional LMDeploy engine dependencies.
    path = Path(__file__).resolve().parents[2] / "xtuner/v1/rl/rollout/lmdeploy.py"
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_get_request_payload"
    )
    namespace = {"copy": copy}
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    request = SimpleNamespace(
        tools=None,
        tool_choice=None,
        sample_params=SampleParams(return_token_ids=return_token_ids),
        message=state.message,
        tokens=state.prompt_ids,
        extra_fields=state.extra_fields,
    )
    worker = SimpleNamespace(
        model_name="test", enable_return_routed_experts=False, _transform_sample_params=lambda params: {}
    )
    return namespace["_get_request_payload"](worker, request)


def teacher_payload(options=None, backend="lmdeploy"):
    client = RolloutTeacherClient.__new__(RolloutTeacherClient)
    client.backend = backend
    client.target_config = SimpleNamespace(uses_topk_targets=False)
    return client._construct_payload(
        [1], [2], logprob_start_len=0, image_data=["test.png"], image_processor_kwargs=options
    )


@pytest.mark.parametrize(
    "options", [{"min_image_tokens": 64, "max_image_tokens": 64}, {"min_pixels": 4096, "max_pixels": 16384}]
)
@pytest.mark.parametrize("return_token_ids", [True, False])
def test_both_payload_paths_forward_options(options, return_token_ids):
    original = dict(options)
    state = RolloutState(message=[], prompt_ids=[1], extra_fields={"image_processor_kwargs": options})
    payload = actor_payload(state, return_token_ids)
    scored = teacher_payload(options)
    assert payload["mm_processor_kwargs"] == scored["mm_processor_kwargs"] == {"image": options}
    assert json.loads(json.dumps(payload))["mm_processor_kwargs"] == {"image": original}
    assert json.loads(json.dumps(scored))["mm_processor_kwargs"] == {"image": original}
    assert options == original
    assert actor_payload(state, return_token_ids)["mm_processor_kwargs"] == {"image": options}


def test_absent_options_preserve_text_requests():
    assert "mm_processor_kwargs" not in actor_payload(RolloutState(message=[], prompt_ids=[1]))
    assert "mm_processor_kwargs" not in teacher_payload()
    assert "mm_processor_kwargs" not in teacher_payload({"min_image_tokens": 64}, backend="sglang")


@pytest.mark.parametrize(
    "family,env,config,budget",
    [
        ("glm", "GLM_5_3_FLASH_PATH", RLGlm53VLTokenizeFnConfig, 64),
        ("qwen", "QWEN3_VL_PATH", RLQwen3VLTokenizeFnConfig, 65536),
    ],
)
@pytest.mark.parametrize("custom", [False, True])
def test_real_producer_and_hf_preprocessing(family, env, config, budget, custom, tmp_path):
    model = os.environ.get(env)
    if not model:
        pytest.skip(f"Set {env} for real CPU processor tests")
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    opts = {"min_pixels": budget, "max_pixels": budget} if custom else {}
    fn = config(processor_path=model, **opts).build(tok)
    fn.set_state("runtime")
    image = tmp_path / "image.png"
    Image.new("RGB", (112, 112), (20, 50, 80)).save(image)
    item = {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": str(image), "image_wh": [112, 112]}},
                    {
                        "type": "text",
                        "text": "<IMG_CONTEXT>\nDescribe this." if family == "qwen" else "Describe this.",
                    },
                ],
            }
        ],
        "data_source": "vl",
        "extra_info": {"tag": 1},
    }
    original = copy.deepcopy(item)
    state = fn(item)
    assert item["extra_info"] == original["extra_info"]
    if family == "glm":
        assert item == original
    options = state.extra_fields["image_processor_kwargs"]
    payload = actor_payload(state)
    assert teacher_payload(options)["mm_processor_kwargs"] == payload["mm_processor_kwargs"]
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    defaults = copy.deepcopy(processor.image_processor.to_dict())
    if family == "glm":
        assert options == {
            "min_image_tokens": fn.image_processor.min_image_tokens,
            "max_image_tokens": fn.image_processor.max_image_tokens,
        }
        kwargs = options
    else:
        assert options == {
            "min_pixels": fn.image_processor.size["shortest_edge"],
            "max_pixels": fn.image_processor.size["longest_edge"],
        }
        kwargs = {"size": {"shortest_edge": options["min_pixels"], "longest_edge": options["max_pixels"]}}
    # Optional fixed upstream helper supplies the exact PR5113 adapter for local integration verification.
    helper_path = os.environ.get("LMDEPLOY_PROCESSOR_UTILS")
    if helper_path:
        import logging

        namespace = {"Any": object, "deepcopy": copy.deepcopy, "logger": logging.getLogger(__name__)}
        functions = [
            node
            for node in ast.parse(Path(helper_path).read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in {"get_override_size", "get_processor_kwargs"}
        ]
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), helper_path, "exec"),
            namespace,
        )
        kwargs = namespace["get_processor_kwargs"](
            processor.image_processor, payload["mm_processor_kwargs"]["image"], "image"
        )
    out = processor(
        text=[state.extra_fields["raw_prompt"]], images=[Image.open(image)], images_kwargs=kwargs, return_tensors="pt"
    )
    assert out["input_ids"][0].tolist() == state.extra_fields["train_prompt_ids"]
    np.testing.assert_array_equal(out["image_grid_thw"].numpy(), state.mm_info["image_grid_thw"])
    np.testing.assert_array_equal(out["pixel_values"].numpy(), state.mm_info["pixel_values"])
    assert processor.image_processor.to_dict() == defaults
    expected = copy.deepcopy(options)
    state.extra_fields["image_processor_kwargs"].clear()
    assert fn(original).extra_fields["image_processor_kwargs"] == expected
