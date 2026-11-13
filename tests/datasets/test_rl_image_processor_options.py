"""CPU image-budget propagation, without starting an inference server."""

import ast
import asyncio
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from PIL import Image

from transformers import AutoProcessor, AutoTokenizer
from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams
from xtuner.v1.datasets.rl_tokenize_fn.glm53_vl_tokenize_fn import RLGlm53VLTokenizeFnConfig
from xtuner.v1.datasets.rl_tokenize_fn.qwen3_vl_tokenize_fn import RLQwen3VLTokenizeFnConfig
from xtuner.v1.rl.distillation.config import RolloutTeacherConfig, TeacherTargetConfig
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


@pytest.mark.parametrize("prefix_caching", [False, True])
@pytest.mark.parametrize(
    "options", [{"min_image_tokens": 64, "max_image_tokens": 64}, {"min_pixels": 65536, "max_pixels": 65536}]
)
def test_teacher_compute_logprobs_sends_image_budgets(options, prefix_caching, monkeypatch):
    monkeypatch.setenv("XTUNER_USE_LMDEPLOY", "1")
    monkeypatch.setenv("XTUNER_USE_SGLANG", "0")
    monkeypatch.setenv("XTUNER_USE_VLLM", "0")
    state = RolloutState(
        message=[],
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        extra_fields={
            "image_data": ["test.png"],
            "image_processor_kwargs": options,
            "train_prompt_ids": [1, 99, 99, 2],
        },
    )
    original = copy.deepcopy(state.extra_fields)
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["input_ids"] == [1, 2, 3, 4]
        assert payload["image_data"] == ["test.png"]
        assert payload["mm_processor_kwargs"] == {"image": options}
        assert payload["logprob_start_len"] == (3 if prefix_caching else 0)
        rows = [[-0.1, 99], [-0.1, 99], [-0.1, 2], [-0.5, 3], [-0.7, 4]]
        return httpx.Response(
            200,
            json={"meta_info": {"prompt_tokens": 6, "input_token_logprobs": rows[-2:] if prefix_caching else rows}},
        )

    async def score():
        client = RolloutTeacherClient(
            RolloutTeacherConfig(
                name="test",
                endpoints=["http://teacher.test"],
                enable_prefix_caching=prefix_caching,
                max_retry_per_sample=0,
            ),
            TeacherTargetConfig(mode="sampled_token"),
        )
        await client.aclose()
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client._client:
            assert await client.compute_logprobs(state) is state

    asyncio.run(score())
    assert len(requests) == 1
    assert state.teacher_targets.tokens == [3, 4]
    assert state.teacher_targets.logprobs == [-0.5, -0.7]
    assert state.extra_fields["teacher_score_time_s"] >= 0
    assert {key: value for key, value in state.extra_fields.items() if key != "teacher_score_time_s"} == original


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


@pytest.fixture(scope="module", params=["glm", "qwen"])
def lmdeploy_frontend(request):
    """Opt-in cross-repository check against an actual LMDeploy checkout.

    Set LMDEPLOY_SOURCE plus both model path variables; no weights or GPU are loaded.
    Unlike the helper-only check above, this imports the serving/frontend implementation.
    """
    source = os.environ.get("LMDEPLOY_SOURCE")
    if not source:
        pytest.skip("Set LMDEPLOY_SOURCE to test LMDeploy serving/preprocessing")
    family = request.param
    env = "GLM_5_3_FLASH_PATH" if family == "glm" else "QWEN3_VL_PATH"
    model_path = os.environ.get(env)
    if not model_path:
        pytest.fail(f"LMDEPLOY_SOURCE integration requires {env}")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(source)
        import lmdeploy
        from lmdeploy.model import HFChatTemplate
        from lmdeploy.serve.processors.multimodal import MultimodalProcessor
        from lmdeploy.vl.engine import ImageEncoder
        from lmdeploy.vl.model.glm5_next import GLM5NextVisionModel
        from lmdeploy.vl.model.qwen3 import Qwen3VLModel

        from transformers import AutoConfig

        assert Path(lmdeploy.__file__).resolve().is_relative_to(Path(source).resolve())
        cls = GLM5NextVisionModel if family == "glm" else Qwen3VLModel
        model = cls(model_path, hf_config=AutoConfig.from_pretrained(model_path, trust_remote_code=True))
        model.build_preprocessor(trust_remote_code=True)
        # Skip only encoder/model-weight initialization; use its real async preprocess dispatch.
        encoder = ImageEncoder.__new__(ImageEncoder)
        encoder.model = model
        encoder.executor = None
        encoder._uses_new_preprocess = encoder._is_new_preprocess_api(model)
        frontend = MultimodalProcessor(
            tokenizer=None,
            chat_template=HFChatTemplate(model_path, trust_remote_code=True),
            vl_encoder=encoder,
            backend="pytorch",
        )
        config = RLGlm53VLTokenizeFnConfig if family == "glm" else RLQwen3VLTokenizeFnConfig
        yield family, model_path, config, model, frontend


async def preprocess_serving_payload(payload, frontend, *, chat):
    """Run the real request schema/route and frontend, stopping before GPU generation."""
    from fastapi import APIRouter
    from lmdeploy.messages import PytorchEngineConfig
    from lmdeploy.pytorch.disagg.config import EngineRole
    from lmdeploy.serve.core.async_engine import AsyncEngine
    from lmdeploy.serve.core.chat_runner import ChatRunner
    from lmdeploy.serve.managers import SessionManager
    from lmdeploy.serve.openai.endpoints.generate import register
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest, GenerateReqInput

    class Preprocessed(Exception):
        def __init__(self, result):
            self.result = result

    engine_config = PytorchEngineConfig(logprobs_mode="raw_logprobs", role=EngineRole.Hybrid)
    sessions = SessionManager()
    # Replace only model/runtime initialization and generation-config normalization.
    # Keep AsyncEngine.preprocess itself so its processor-argument forwarding is checked.
    engine = SimpleNamespace(
        arch="Glm5NextForConditionalGeneration"
        if frontend.vl_encoder.model.__class__.__name__ == "GLM5NextVisionModel"
        else "Qwen3VLForConditionalGeneration",
        session_mgr=sessions,
        prompt_processor=frontend,
        session_len=100000,
        backend_config=engine_config,
        request_logger=SimpleNamespace(
            log_prompt=lambda *args, **kwargs: None, log_inputs=lambda *args, **kwargs: None
        ),
        _determine_gen_config=lambda input_ids, gen_config: gen_config,
    )

    async def preprocess(messages=None, session_id=None, **kwargs):
        assert kwargs.get("input_ids") is None
        result = await AsyncEngine.preprocess(engine, messages, session_id, **kwargs)
        raise Preprocessed(result.inputs)

    context = SimpleNamespace(
        async_engine=SimpleNamespace(model_name="test", preprocess=preprocess),
        engine_config=engine_config,
        session_manager=SimpleNamespace(has=lambda _: False),
        create_session=lambda _: sessions.get(1),
        default_gen_config={},
        response_parser_cls=lambda req: SimpleNamespace(request=req),
    )
    # Exercise the wire representation and Pydantic schema, rather than sharing Python dicts.
    body = json.dumps(payload)
    with pytest.raises(Preprocessed) as caught:
        if chat:
            await ChatRunner.prepare(context, ChatCompletionRequest.model_validate_json(body))
        else:
            router = APIRouter()
            register(router, context)
            endpoint = next(route.endpoint for route in router.routes if route.path == "/generate")
            await endpoint(GenerateReqInput.model_validate_json(body))
    return caught.value.result


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("image_count", [1, 2])
def test_lmdeploy_serving_consumes_tokenizer_budgets(lmdeploy_frontend, custom, image_count, tmp_path):
    family, model_path, config, model, frontend = lmdeploy_frontend
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    budget = 64 if family == "glm" else 65536
    fn = config(processor_path=model_path, **({"min_pixels": budget, "max_pixels": budget} if custom else {})).build(
        tok
    )
    fn.set_state("runtime")
    contents = []
    for index, (size, color) in enumerate([((448, 336), (20, 50, 80)), ((224, 448), (160, 70, 30))][:image_count]):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", size, color).save(path)
        contents.append({"type": "image_url", "image_url": {"url": str(path), "image_wh": list(size)}})
    text = "<IMG_CONTEXT>\n" * image_count if family == "qwen" else ""
    contents.append({"type": "text", "text": text + "Describe these images."})
    state = fn({"prompt": [{"role": "user", "content": contents}], "data_source": "vl"})
    original = copy.deepcopy(state.extra_fields["image_processor_kwargs"])
    defaults = copy.deepcopy(model.processor.image_processor.to_dict())
    response_ids = tok.encode("Answer", add_special_tokens=False)
    client = RolloutTeacherClient.__new__(RolloutTeacherClient)
    client.backend = "lmdeploy"
    client.target_config = SimpleNamespace(uses_topk_targets=False)
    teacher = client._construct_payload(
        state.prompt_ids,
        response_ids,
        logprob_start_len=0,
        image_data=state.extra_fields["image_data"],
        image_processor_kwargs=original,
    )
    teacher["model"] = "test"

    async def check_requests():
        # Consecutive calls share the serving processor, exposing request-default pollution.
        for payload, chat, suffix in [
            (actor_payload(state), False, []),
            (actor_payload(state, return_token_ids=False), True, None),
            (teacher, False, response_ids),
        ]:
            result = await preprocess_serving_payload(payload, frontend, chat=chat)
            if suffix is not None:
                assert result["input_ids"] == state.extra_fields["train_prompt_ids"] + suffix
            grids = np.stack([part["image_grid_thw"].numpy() for part in result["multimodal"]])
            pixels = np.concatenate([part["pixel_values"].numpy() for part in result["multimodal"]])
            np.testing.assert_array_equal(grids, state.mm_info["image_grid_thw"])
            np.testing.assert_array_equal(pixels, state.mm_info["pixel_values"])
            assert model.processor.image_processor.to_dict() == defaults
            assert state.extra_fields["image_processor_kwargs"] == original

        # A following request with no overrides must recover the server's original defaults.
        without_options = actor_payload(state)
        without_options.pop("mm_processor_kwargs")
        restored = await preprocess_serving_payload(without_options, frontend, chat=False)
        expected = model.processor(
            text=[state.extra_fields["raw_prompt"]],
            images=[Image.open(part["image_url"]["url"]) for part in contents[:-1]],
            return_tensors="pt",
        )
        np.testing.assert_array_equal(
            np.stack([part["image_grid_thw"].numpy() for part in restored["multimodal"]]),
            expected["image_grid_thw"].numpy(),
        )
        assert model.processor.image_processor.to_dict() == defaults

    asyncio.run(check_requests())
