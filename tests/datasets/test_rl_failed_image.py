"""Bad local images become failed RL inputs before generation, not fake prompts."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.datasets.mllm_tokenize_fn.glm53_vl_tokenize_fn import Glm53VLTokenizeFunction
from xtuner.v1.datasets.mllm_tokenize_fn.qwen3_vl_tokenize_fn import Qwen3VLTokenizeFunction
from xtuner.v1.datasets.rl_tokenize_fn.glm53_vl_tokenize_fn import (
    RLGlm53VLTokenizeFnConfig,
    RLGlm53VLTokenizeFunction,
)
from xtuner.v1.datasets.rl_tokenize_fn.qwen3_vl_tokenize_fn import RLQwen3VLTokenizeFunction
from xtuner.v1.datasets.vlm_jsonl import VLMJsonlDataset
from xtuner.v1.rl.agent_loop_manager import (
    AsyncProduceStrategyConfig,
    ProduceContext,
    ProduceProgress,
    SyncProduceStrategyConfig,
)
from xtuner.v1.rl.agent_loop_manager.sampler import Sampler
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig


ADAPTERS = [
    (RLGlm53VLTokenizeFunction, Glm53VLTokenizeFunction),
    (RLQwen3VLTokenizeFunction, Qwen3VLTokenizeFunction),
]


def image_item(path):
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": {"url": path, "image_wh": [112, 112]}},
                    {"type": "text", "text": "Describe this."},
                ],
            }
        ],
        "data_source": "vl",
    }


def adapter_stub(cls):
    fn = cls.__new__(cls)
    fn.state = "runtime"
    fn.system_message = None
    fn.data_judger_mapping = {"vl": {"judge": 0.5}}
    return fn


@pytest.mark.parametrize("cls,parent", ADAPTERS)
@pytest.mark.parametrize("damage", ["missing", "unidentified"])
def test_adapters_convert_real_local_image_errors(tmp_path, monkeypatch, cls, parent, damage):
    path = tmp_path / "bad.png"
    if damage == "unidentified":
        path.write_bytes(b"not an image")

    def read_image(self, item, media_root=""):
        self._image_path = [path.name]
        with Image.open(os.path.join(media_root, path.name)) as image:
            image.convert("RGB")
        raise AssertionError("Expected image loading to fail")

    monkeypatch.setattr(parent, "__call__", read_image)
    item = image_item(path.name)
    state = adapter_stub(cls)(item, media_root=str(tmp_path))
    assert state.status == Status.FAILED
    assert state.message == item["prompt"]
    assert state.data_source == {"judge": 0.5}
    assert str(path) in state.error_msg
    assert state.mm_info is state.prompt_ids is state.input_ids is state.labels is None
    assert state.extra_fields == {}


@pytest.mark.parametrize("cls,parent", ADAPTERS)
@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(2, "missing processor config", "/unrelated/processor.json"),
        UnidentifiedImageError("cannot identify image file <_io.BytesIO object>"),
        OSError("decoder or storage error without an identified source"),
        ValueError("invalid processor configuration"),
        RuntimeError("programming error"),
    ],
)
def test_adapters_do_not_swallow_unrelated_errors(monkeypatch, cls, parent, error):
    def fail(self, *args, **kwargs):
        self._image_path = ["sample.png"]
        raise error

    monkeypatch.setattr(parent, "__call__", fail)
    with pytest.raises(type(error)) as caught:
        adapter_stub(cls)(image_item("sample.png"))
    assert caught.value is error


@pytest.mark.parametrize("cls,parent", ADAPTERS)
def test_cache_does_not_return_rollout_failure(monkeypatch, cls, parent):
    error = FileNotFoundError(2, "missing", "/missing.png")

    def fail(self, *args, **kwargs):
        self._image_path = ["/missing.png"]
        raise error

    monkeypatch.setattr(parent, "__call__", fail)
    fn = adapter_stub(cls)
    fn.state = "cache"
    with pytest.raises(FileNotFoundError) as caught:
        fn(image_item("/missing.png"))
    assert caught.value is error


@pytest.mark.parametrize("cls,parent", ADAPTERS)
@pytest.mark.parametrize("missing_key", ["prompt", "data_source"])
def test_invalid_annotation_is_not_a_failed_media_sample(monkeypatch, cls, parent, missing_key):
    def fail(self, *args, **kwargs):
        self._image_path = ["/missing.png"]
        raise FileNotFoundError(2, "missing image", "/missing.png")

    monkeypatch.setattr(parent, "__call__", fail)
    item = image_item("/missing.png")
    del item[missing_key]
    with pytest.raises((KeyError, AssertionError)):
        adapter_stub(cls)(item)


@pytest.fixture(scope="module")
def real_glm_tokenizer():
    path = os.environ.get("GLM_5_3_FLASH_PATH")
    if not path:
        pytest.skip("Set GLM_5_3_FLASH_PATH for real processor coverage")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    return RLGlm53VLTokenizeFnConfig(processor_path=path, ignore_multimodal_info=True).build(tokenizer)


@pytest.mark.parametrize("config", [SyncProduceStrategyConfig(), AsyncProduceStrategyConfig()])
def test_real_bad_images_are_discarded_and_replaced_before_rollout(tmp_path, real_glm_tokenizer, config):
    (tmp_path / "broken.png").write_bytes(b"not an image")
    Image.new("RGB", (112, 112), (30, 60, 90)).save(tmp_path / "valid.png")
    items = [image_item(name) for name in ("missing.png", "broken.png", "valid.png")]
    lines = [json.dumps(item) + "\n" for item in items]
    path = tmp_path / "samples.jsonl"
    path.write_text("".join(lines))
    dataset = VLMJsonlDataset.__new__(VLMJsonlDataset)
    dataset.path = str(path)
    dataset._meta = {"offsets": [0, len(lines[0]), len(lines[0]) + len(lines[1])]}
    dataset.media_root = str(tmp_path)
    dataset.tokenize_fn = real_glm_tokenizer
    real_glm_tokenizer.set_state("runtime")
    # 无 fake_data，且有效图只在本测试关闭 mm_info 传输，避免初始化真实 Ray object store。
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda samples: samples, num_workers=0)

    async def run():
        buffer = AsyncReplayBufferConfig().build()
        sampler = Sampler(loader, prompt_repeat_k=2, replay_buffer=buffer)

        async def generate(group, **kwargs):
            for item in group:
                assert item.status == Status.INIT
                assert item.prompt_ids
                assert item.extra_fields["image_data"] == [str(tmp_path / "valid.png")]
                item.status = Status.COMPLETED
                item.response_ids = []
            return group

        agent_loop = Mock(generate_group=AsyncMock(side_effect=generate))
        progress = ProduceProgress.build(task_names=["vl"], target_samples={"vl": 1})
        context = ProduceContext(
            agent_loop=agent_loop,
            sampler=sampler,
            replay_buffer=buffer,
            task_batch_size=1,
            task_name="vl",
            train_step=0,
            model_step=0,
            progress=progress,
        )
        await asyncio.wait_for(config.build().produce_batch(context), timeout=10)
        groups = await buffer.get(1, task_name="vl", group_status=Status.COMPLETED)
        assert len(groups) == 1 and len(groups[0]) == 2
        assert groups[0][0].group_id == 2
        assert progress.failed_samples["vl"] == 4
        agent_loop.generate_group.assert_awaited_once()

    asyncio.run(run())
