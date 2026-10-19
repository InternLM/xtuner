"""GLM-5.3-Flash 的 VL TokenizeFn（cache/runtime 双路径），见设计文档 F1。

TestGlm53TokenizeCacheRuntimeParity
    test_glm53_tokenize_cache_runtime_image_parity  图像：cache 预测与 runtime 实算一致
    test_glm53_tokenize_cache_runtime_video_parity  视频：同上
    test_glm53_frame_folder_samples_the_processed_frames  帧文件夹按重采样后的帧取下标
TestGlm53VisualPlaceholderAndTokenSemantics
    test_glm53_visual_placeholder_count             占位符数量等于 patch 数 / merge_unit
    test_glm53_num_img_tokens_is_per_vit_sequence   num_img_tokens 按 ViT 序列计数
    test_glm53_mm_token_type_ids_alignment          mm_token_type_ids 与 input_ids 等长对齐
TestGlm53MmTokenTypeIdsMatchesProcessorGolden
    test_image_only                                 图像模态标记与 HF processor 一致
    test_video_only                                 视频模态标记与 HF processor 一致
TestGlm53NativeHFInputContract
    test_full_image_input_matches_native_template   完整图像输入与原生模板/processor 一致
    test_enable_thinking_false_rejected             enable_thinking=False 被显式拒绝（jinja 忽略该参数）
    test_video_pixels_and_timestamps_match_native_template  视频像素/时间戳与原生模板一致
TestGlm53MixedMediaAndTruncation
    test_glm53_mixed_media_fails_fast               图文混合视频未支持，必须快速失败
    test_glm53_visual_truncation_is_dropped         截断会切断视觉跨度时 cache 阶段就丢弃
TestGlm53MediaRoleContract
    test_media_is_collected_from_all_roles           system/assistant/tool 角色的图片也会被收集
    test_media_collection_preserves_input_order     媒体按原始消息顺序收集，与 HF 一致
    test_string_content_parts_are_tolerated         content 列表混排纯字符串不崩溃
TestGlm53CacheInvalidation
    test_glm53_cache_invalidates_on_processor_or_pack_change  processor/pack 变化使缓存失效
    test_glm53_hash_covers_every_init_parameter  逐参数 hash 失效 + __init__ 参数分类完备性
"""

import copy
import inspect
import os
import weakref
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase
from transformers.video_utils import VideoMetadata
from xtuner.v1.datasets.mllm_tokenize_fn import Glm53VLTokenizeFnConfig, Glm53VLTokenizeFunction
from xtuner.v1.datasets.mllm_tokenize_fn.base_mllm_tokenize_fn import (
    BaseMLLMTokenizeFunction,
    collect_image_video_paths_and_extra,
)
from xtuner.v1.datasets.mllm_tokenize_fn.qwen3_vl_utils import sort_frames


GLM_5_3_FLASH_PATH = os.environ["GLM_5_3_FLASH_PATH"]


@pytest.fixture(scope="module")
def ckpt_path():
    if not os.path.isdir(GLM_5_3_FLASH_PATH):
        pytest.fail(f"GLM_5_3_FLASH_PATH not found: {GLM_5_3_FLASH_PATH}")
    return GLM_5_3_FLASH_PATH


@pytest.fixture(scope="module")
def tokenizer(ckpt_path):
    return AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)


@pytest.fixture(scope="module")
def processor(ckpt_path):
    return AutoProcessor.from_pretrained(ckpt_path)


@pytest.fixture()
def tokenize_fn(ckpt_path, tokenizer):
    return Glm53VLTokenizeFnConfig(processor_path=ckpt_path).build(tokenizer, anno_name="test")


@pytest.fixture(scope="module")
def image_fixture(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm53_img") / "img.png"
    Image.new("RGB", (448, 448), color=(120, 30, 200)).save(path)
    return str(path)


@pytest.fixture(scope="module")
def video_fixture(tmp_path_factory):
    video_dir = tmp_path_factory.mktemp("glm53_video") / "video1"
    video_dir.mkdir()
    for i in range(6):
        Image.fromarray(np.full((224, 224, 3), i * 30, dtype=np.uint8)).save(video_dir / f"{i + 1:08d}.jpg")
    return str(video_dir)


def _image_item(image_path: str, width: int = 448, height: int = 448):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": {"url": image_path, "image_wh": [width, height]}},
                    {"type": "text", "text": "what is this?"},
                ],
            },
            {"role": "assistant", "content": "a color swatch"},
        ]
    }


def _video_item(
    video_dir: str,
    num_frames: int = 6,
    fps: float = 2.0,
    width: int = 224,
    height: int = 224,
    processed: tuple[int, float] | None = None,
):
    extra = {} if processed is None else {"processed_video_length": processed[0], "processed_fps": processed[1]}
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "url": video_dir,
                            "image_wh": [width, height],
                            "origin_video_length": num_frames,
                            "origin_fps": fps,
                            **extra,
                        },
                    },
                    {"type": "text", "text": "describe"},
                ],
            },
            {"role": "assistant", "content": "a gradient video"},
        ]
    }


class TestGlm53TokenizeCacheRuntimeParity:
    def test_glm53_tokenize_cache_runtime_image_parity(self, tokenize_fn, image_fixture):
        item = _image_item(image_fixture)
        tokenize_fn.state = "cache"
        cache_ret = tokenize_fn(item)
        tokenize_fn.state = "runtime"
        runtime_ret = tokenize_fn(item)

        assert cache_ret["num_tokens"] == runtime_ret["num_tokens"] == len(runtime_ret["input_ids"])
        assert cache_ret["num_img_tokens"] == runtime_ret["num_img_tokens"]

    def test_glm53_tokenize_cache_runtime_video_parity(self, tokenize_fn, video_fixture):
        item = _video_item(video_fixture)
        tokenize_fn.state = "cache"
        cache_ret = tokenize_fn(item)
        tokenize_fn.state = "runtime"
        runtime_ret = tokenize_fn(item)

        assert cache_ret["num_tokens"] == runtime_ret["num_tokens"] == len(runtime_ret["input_ids"])
        assert cache_ret["num_img_tokens"] == runtime_ret["num_img_tokens"]

    def test_glm53_frame_folder_samples_the_processed_frames(self, tokenize_fn, video_fixture):
        # 帧文件夹里是重采样之后的视频（此处 6 帧 @2fps），元数据却同时描述了更长的原始视频
        # （180 帧 @30fps）——真实 VLM 数据就是这个形态。之前按原始视频算帧下标，runtime 直接越界，
        # 数据集静默换成 33 token 的假样本，而 cache 仍按原始视频预测长度，打包按错误长度装箱。
        item = _video_item(video_fixture.rstrip("/") + "/", num_frames=180, fps=30.0, processed=(6, 2.0))
        tokenize_fn.state = "cache"
        cache_ret = tokenize_fn(item)
        tokenize_fn.state = "runtime"
        runtime_ret = tokenize_fn(item)

        assert cache_ret["num_tokens"] == runtime_ret["num_tokens"] == len(runtime_ret["input_ids"])
        assert cache_ret["num_img_tokens"] == runtime_ret["num_img_tokens"]


class TestGlm53OSSLoader:
    def test_oss_image_matches_local_and_cache(
        self,
        ckpt_path: str,
        tokenizer: PreTrainedTokenizerBase,
        image_fixture: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = Mock()
        backend = Mock(return_value=client)
        monkeypatch.setattr("xtuner.v1.datasets.mllm_tokenize_fn.qwen3_vl_utils.get_oss_backend", backend)
        fn = Glm53VLTokenizeFnConfig(
            processor_path=ckpt_path,
            oss_loader_cfg={"backend": "petrel", "backend_kwargs": {"conf_path": "test.conf"}},
        ).build(tokenizer, anno_name="test")
        backend.assert_called_once_with("petrel", conf_path="test.conf")

        local_item = _image_item(image_fixture)
        remote_item = _image_item("img.png")
        client.get.side_effect = {"s3://bucket/img.png": Path(image_fixture).read_bytes()}.__getitem__
        fn.state = "cache"
        cache = fn(remote_item, media_root="s3://bucket/")
        client.get.assert_not_called()
        fn.state = "runtime"
        local = fn(local_item)
        client.get.assert_not_called()
        remote = fn(remote_item, media_root="s3://bucket/")
        assert client.get.called
        assert cache["num_tokens"] == remote["num_tokens"] == local["num_tokens"]
        assert cache["num_img_tokens"] == remote["num_img_tokens"] == local["num_img_tokens"]
        for key in ("input_ids", "labels"):
            assert remote[key] == local[key]
        for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            assert torch.equal(remote[key], local[key])

    def test_oss_image_requires_loader(self, tokenize_fn: Glm53VLTokenizeFunction) -> None:
        tokenize_fn.state = "runtime"
        with pytest.raises(AssertionError, match="oss_loader_cfg"):
            tokenize_fn(_image_item("s3://bucket/image.png"))

    def test_oss_loader_cfg_does_not_change_cache_hash(
        self, ckpt_path: str, tokenizer: PreTrainedTokenizerBase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The media backend only affects runtime reading, never cache-path geometry:
        # a differing hash would silently invalidate every cached length.
        monkeypatch.setattr(
            "xtuner.v1.datasets.mllm_tokenize_fn.qwen3_vl_utils.get_oss_backend", Mock(return_value=Mock())
        )
        plain = Glm53VLTokenizeFnConfig(processor_path=ckpt_path).build(tokenizer, anno_name="test")
        with_oss = Glm53VLTokenizeFnConfig(
            processor_path=ckpt_path,
            oss_loader_cfg={"backend": "petrel", "backend_kwargs": {"conf_path": "test.conf"}},
        ).build(tokenizer, anno_name="test")
        assert plain.hash() == with_oss.hash()


class TestFrameNamingConvention:
    def test_sort_frames_rejects_nonconforming_names(self) -> None:
        with pytest.raises(ValueError, match="naming convention"):
            sort_frames(["cluster:s3://bucket/frames/frame.0068.tonemap.jpg"])


class TestGlm53VisualPlaceholderAndTokenSemantics:
    def test_glm53_visual_placeholder_count(self, tokenize_fn, image_fixture):
        tokenize_fn.state = "runtime"
        ret = tokenize_fn(_image_item(image_fixture))
        image_token_id = tokenize_fn.processor.image_token_id
        placeholder_count = ret["input_ids"].count(image_token_id)
        merge_unit = tokenize_fn.merge_unit
        assert placeholder_count == sum(ret["num_img_tokens"]) // merge_unit

    def test_glm53_num_img_tokens_is_per_vit_sequence(self, tokenize_fn, video_fixture):
        tokenize_fn.state = "runtime"
        ret = tokenize_fn(_video_item(video_fixture))
        grid_t, grid_h, grid_w = ret["video_grid_thw"][0].tolist()
        assert ret["num_img_tokens"] == [grid_h * grid_w] * grid_t
        assert len(ret["num_img_tokens"]) == grid_t  # not one merged t*h*w entry

    def test_glm53_mm_token_type_ids_alignment(self, tokenize_fn, image_fixture):
        tokenize_fn.state = "runtime"
        ret = tokenize_fn(_image_item(image_fixture))
        assert ret["mm_token_type_ids"].shape[-1] == len(ret["input_ids"])
        image_token_id = tokenize_fn.processor.image_token_id
        is_placeholder = torch.tensor([tid == image_token_id for tid in ret["input_ids"]])
        assert (ret["mm_token_type_ids"][0] == 1).equal(is_placeholder & (ret["mm_token_type_ids"][0] == 1))
        assert (ret["mm_token_type_ids"][0][is_placeholder] == 1).all()
        assert (ret["mm_token_type_ids"][0][~is_placeholder] == 0).all()


class TestGlm53MmTokenTypeIdsMatchesProcessorGolden:
    def test_image_only(self, tokenize_fn, processor, image_fixture):
        item = _image_item(image_fixture)
        tokenize_fn.state = "runtime"
        ret = tokenize_fn(item)
        text, _ = tokenize_fn._render_base(item)
        golden = processor(text=[text], images=[Image.open(image_fixture).convert("RGB")], return_tensors="pt")

        assert golden["input_ids"][0].tolist() == ret["input_ids"]
        assert golden["mm_token_type_ids"][0].tolist() == ret["mm_token_type_ids"][0].tolist()
        assert (golden["mm_token_type_ids"][0] == 2).sum().item() == 0  # image-only: no type-2

    def test_video_only(self, tokenize_fn, processor, video_fixture):
        item = _video_item(video_fixture)
        tokenize_fn.state = "runtime"
        ret = tokenize_fn(item)
        text, _ = tokenize_fn._render_base(item)

        frames = [Image.open(os.path.join(video_fixture, f)).convert("RGB") for f in sorted(os.listdir(video_fixture))]
        video_tensor = torch.from_numpy(np.stack([np.array(f) for f in frames])).permute(0, 3, 1, 2)
        golden = processor(
            text=[text],
            videos=[video_tensor],
            video_metadata=[VideoMetadata(total_num_frames=6, fps=2.0)],
            return_tensors="pt",
            do_sample_frames=True,
        )

        assert golden["input_ids"][0].tolist() == ret["input_ids"]
        assert golden["mm_token_type_ids"][0].tolist() == ret["mm_token_type_ids"][0].tolist()
        assert (golden["mm_token_type_ids"][0] == 1).sum().item() == 0  # video-only: no type-1


class TestGlm53NativeHFInputContract:
    @pytest.mark.parametrize("media_count", [0, 1, 2])
    @pytest.mark.parametrize("generation", [False, True])
    def test_full_image_input_matches_native_template(
        self, tokenize_fn, processor, image_fixture, media_count: int, generation: bool
    ) -> None:
        item = _image_item(image_fixture)
        media = item["messages"][0]["content"][0]
        item["messages"][0]["content"] = [
            {"type": "text", "text": "比较图片：\n"},
            *[copy.deepcopy(media) for _ in range(media_count)],
            {"type": "text", "text": "\n请说明。"},
        ]
        if generation:
            item["messages"].pop()
        else:
            item["messages"] += [
                {"role": "user", "content": "再说一次"},
                {"role": "assistant", "content": "紫色。"},
            ]
        tokenize_fn.add_generation_prompt = generation
        tokenize_fn.state = "runtime"
        actual = tokenize_fn(item)
        text = processor.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=generation,
            enable_thinking=True,
            reasoning_effort="max",
            clear_thinking=False,
        )
        # XTuner supervises the terminal stop token; HF renders the inference conversation without it.
        if not generation:
            text += processor.tokenizer.eos_token
        images = [Image.open(image_fixture).convert("RGB") for _ in range(media_count)]
        golden = processor(text=[text], images=images or None, return_tensors="pt", add_special_tokens=False)
        assert actual["input_ids"] == golden["input_ids"][0].tolist()
        assert torch.equal(actual["mm_token_type_ids"], golden["mm_token_type_ids"])
        if media_count:
            assert torch.equal(actual["image_grid_thw"], golden["image_grid_thw"])
            assert torch.equal(actual["pixel_values"], golden["pixel_values"])
        assert actual["labels"][0] == -100
        for index, token in enumerate(actual["input_ids"]):
            if token == processor.image_token_id:
                assert actual["labels"][index] == -100
        tokenize_fn.state = "cache"
        cached = tokenize_fn(item)
        assert cached["num_tokens"] == len(actual["input_ids"])
        assert cached["num_img_tokens"] == actual["num_img_tokens"]

    def test_enable_thinking_false_rejected(self, tokenize_fn, image_fixture) -> None:
        # 官方模板忽略 enable_thinking；False 会被静默忽略，xtuner 侧显式拒绝（cache/runtime 同）。
        tokenize_fn.enable_thinking = False
        item = _image_item(image_fixture)
        for state in ("runtime", "cache"):
            tokenize_fn.state = state
            with pytest.raises(AssertionError, match="enable_thinking"):
                tokenize_fn(item)

    @pytest.mark.parametrize("source", ["padded", "numbered", "file"])
    def test_video_pixels_and_timestamps_match_native_template(
        self, tokenize_fn, processor, tmp_path: Path, source: str
    ) -> None:
        frames = []
        for index in range(12):
            frame = Image.new("RGB", (112, 112), (index * 20, 10, 200))
            name = f"frame_{index}.jpg" if source == "numbered" else f"{index:08d}.jpg"
            frame.save(tmp_path / name)
            frames.append(np.array(Image.open(tmp_path / name).convert("RGB")))
        video_path = str(tmp_path) + "/"
        if source == "file":
            import cv2

            from transformers.video_utils import load_video

            video_path = str(tmp_path / "video.mp4")
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (112, 112))
            assert writer.isOpened()
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            frames, _ = load_video(video_path, backend="decord")
        item = _video_item(video_path, num_frames=12, width=112, height=112)
        tokenize_fn.state = "runtime"
        actual = tokenize_fn(item)
        text = processor.apply_chat_template(item["messages"], tokenize=False) + processor.tokenizer.eos_token
        golden = processor(
            text=[text],
            videos=[np.stack(frames)],
            video_metadata=[VideoMetadata(total_num_frames=12, fps=2.0)],
            do_sample_frames=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        assert actual["input_ids"] == golden["input_ids"][0].tolist()
        assert torch.equal(actual["mm_token_type_ids"], golden["mm_token_type_ids"])
        assert torch.equal(actual["video_grid_thw"], golden["video_grid_thw"])
        assert torch.equal(actual["pixel_values_videos"], golden["pixel_values_videos"])
        tokenize_fn.state = "cache"
        assert tokenize_fn(item)["num_tokens"] == actual["num_tokens"]


class TestGlm53MixedMediaAndTruncation:
    @pytest.mark.parametrize("state", ["cache", "runtime"])
    def test_glm53_mixed_media_fails_fast(
        self, tokenize_fn: Glm53VLTokenizeFunction, state: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert Glm53VLTokenizeFunction.__call__ is BaseMLLMTokenizeFunction.__call__
        load = Mock(side_effect=AssertionError("Mixed samples must fail before loading media"))
        monkeypatch.setattr("xtuner.v1.datasets.mllm_tokenize_fn.glm53_vl_tokenize_fn.load_image", load)
        item = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": {"url": "unused.jpg", "image_wh": [448, 448]}},
                        {
                            "type": "video",
                            "video": {
                                "url": "unused.mp4",
                                "image_wh": [224, 224],
                                "origin_video_length": 6,
                                "origin_fps": 2.0,
                            },
                        },
                    ],
                }
            ]
        }
        tokenize_fn.state = state
        with pytest.raises(NotImplementedError, match="Mixed image and video"):
            tokenize_fn(item)
        load.assert_not_called()

    def test_glm53_visual_truncation_is_dropped(self, ckpt_path, tokenizer, image_fixture):
        # max_length 短于样本、会切进 256 token 的图像跨度：cache 阶段就必须报 num_tokens=0，让
        # 数据集在打包前把它滤掉。只在 runtime 报错是不够的——数据集会吞掉异常换成假样本，而
        # packer 早已按 cache 报的截断长度给它留了位置。
        short_fn = Glm53VLTokenizeFnConfig(processor_path=ckpt_path, max_length=20).build(tokenizer, anno_name="test")
        short_fn.state = "cache"
        assert short_fn(_image_item(image_fixture))["num_tokens"] == 0
        # runtime 仍然拒绝，作为 cache 过滤漏掉时的最后一道防线。
        short_fn.state = "runtime"
        with pytest.raises(AssertionError):
            short_fn(_image_item(image_fixture))


class TestGlm53MediaRoleContract:
    @pytest.mark.parametrize("role", ["system", "assistant", "tool"])
    def test_media_is_collected_from_all_roles(self, tokenize_fn: Glm53VLTokenizeFunction, role: str) -> None:
        image = {"type": "image", "image": {"url": "unused.jpg", "image_wh": [448, 448]}}
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        if role == "system":
            messages.insert(0, {"role": "system", "content": [image, {"type": "text", "text": "policy"}]})
        elif role == "assistant":
            messages[1]["content"] = [image, {"type": "text", "text": "a"}]
        else:
            messages.insert(1, {"role": "tool", "content": [image], "tool_call_id": "t1"})

        image_paths, video_paths, extra = collect_image_video_paths_and_extra(
            messages, allow_all_roles=tokenize_fn.allow_all_roles
        )

        assert image_paths == ["unused.jpg"]
        assert video_paths == []
        assert extra["image_wh"] == [[448, 448]]

    def test_media_collection_preserves_input_order(self, tokenize_fn: Glm53VLTokenizeFunction) -> None:
        def image(url: str) -> dict:
            return {"type": "image", "image": {"url": url, "image_wh": [448, 448]}}

        messages = [
            {"role": "system", "content": [image("system.jpg")]},
            {"role": "user", "content": [image("user.jpg")]},
            {
                "role": "assistant",
                "content": [image("assistant.jpg")],
                "tool_calls": [{"id": "a"}, {"id": "b"}],
            },
            {"role": "tool", "content": [image("tool-b.jpg")], "tool_call_id": "b"},
            {"role": "tool", "content": [image("tool-a.jpg")], "tool_call_id": "a"},
        ]

        image_paths, _, _ = collect_image_video_paths_and_extra(messages, allow_all_roles=tokenize_fn.allow_all_roles)

        # HF apply_chat_template also collects media in input order before rendering the Jinja template.
        assert image_paths == ["system.jpg", "user.jpg", "assistant.jpg", "tool-b.jpg", "tool-a.jpg"]

    def test_string_content_parts_are_tolerated(self, tokenize_fn: Glm53VLTokenizeFunction) -> None:
        # content 列表允许纯字符串与 text 字典混排（官方 jinja 按文本原样渲染）；media collector
        # 不得对字符串块取 c["type"] 而崩溃（飞书报告 E01）。
        item = {
            "messages": [
                {"role": "user", "content": ["prefix", {"type": "text", "text": "正文"}, "suffix"]},
                {"role": "assistant", "content": "ASSISTANT_ANSWER"},
            ]
        }
        tokenize_fn.state = "cache"
        cache_ret = tokenize_fn(item)
        tokenize_fn.state = "runtime"
        runtime_ret = tokenize_fn(item)
        assert cache_ret["num_tokens"] == runtime_ret["num_tokens"] == len(runtime_ret["input_ids"])


class TestGlm53CacheInvalidation:
    def test_glm53_cache_invalidates_on_processor_or_pack_change(self, ckpt_path, tokenizer):
        base = Glm53VLTokenizeFnConfig(processor_path=ckpt_path).build(tokenizer, anno_name="test")
        changed_pixels = Glm53VLTokenizeFnConfig(processor_path=ckpt_path, max_pixels=1000).build(
            tokenizer, anno_name="test"
        )
        changed_fps = Glm53VLTokenizeFnConfig(processor_path=ckpt_path, fps=4.0).build(tokenizer, anno_name="test")
        changed_weight = Glm53VLTokenizeFnConfig(processor_path=ckpt_path, visual_pack_weight=0.5).build(
            tokenizer, anno_name="test"
        )

        assert base.hash() != changed_pixels.hash()
        assert base.hash() != changed_fps.hash()
        assert base.hash() != changed_weight.hash()

    def test_glm53_hash_covers_every_init_parameter(self, ckpt_path, tokenizer):
        params = set(inspect.signature(Glm53VLTokenizeFunction.__init__).parameters) - {"self"}
        classified = _HASH_PARAMS | _NON_HASH_PARAMS
        assert params == classified, (
            f"Glm53VLTokenizeFunction.__init__ params {sorted(params)} != classified "
            f"{sorted(classified)}; unclassified: {sorted(params ^ classified)}. If a new param "
            "can change cache output (num_tokens), add it to _hash_str and _HASH_PARAMS; "
            "otherwise add it to _NON_HASH_PARAMS with a reason."
        )
        assert set(_ALT_VALUES) == _HASH_PARAMS - {"processor_path"}

        for name, override in _ALT_VALUES.items():
            base = Glm53VLTokenizeFnConfig(processor_path=ckpt_path).build(tokenizer, anno_name="test")
            variant = Glm53VLTokenizeFnConfig(processor_path=ckpt_path, **override).build(tokenizer, anno_name="test")
            assert base.hash() != variant.hash(), f"{name} changed but hash() did not -- stale cache entry"


# __init__ 参数中影响 cache 输出（每条样本 num_tokens）的，必须进 _hash_str。processor_path
# 也在 _hash_str 里，但换一个值需要磁盘上真实存在另一个 checkpoint，这里不实际变它。
_HASH_PARAMS = {
    "processor_path",
    "min_pixels",
    "max_pixels",
    "fps",
    "max_frames",
    "system_message",
    "max_length",
    "llm_pack_weight",
    "visual_pack_weight",
    "add_generation_prompt",
    "enable_thinking",
    "reasoning_effort",
}

# 不影响 cache 输出的参数：
#   tokenizer           -- 身份由基类 hash() 里的 tokenizer_xxhash 单独覆盖
#   anno_name           -- 数据文件本身；cache 目录按数据文件隔离，不跨文件共享
#   tokenizer_hash/hash -- 对 hash 结果的显式覆盖
#   debug / oss_time_log_thr / trim_memory_interval -- 运行期开关，不改变 token 数
#   oss_loader_cfg      -- 只控制 runtime 存储读取，另有 cache hash 一致性测试
_NON_HASH_PARAMS = {
    "tokenizer",
    "anno_name",
    "tokenizer_hash",
    "hash",
    "debug",
    "oss_time_log_thr",
    "trim_memory_interval",
    "oss_loader_cfg",
}

# _HASH_PARAMS（除 processor_path）逐参数构造变体用的 override。取值必须避开 processor 默认值
# （min=16 / max=8000 / fps=2 / max_frames=2048）和其他测试残留进共享 processor 的值
# （max_pixels=1000、fps=4.0）。
_ALT_VALUES = {
    "min_pixels": {"min_pixels": 1000},
    "max_pixels": {"max_pixels": 2000},
    "fps": {"fps": 8.0},
    "max_frames": {"max_frames": 8},
    "system_message": {"system_message": "You are a helpful assistant."},
    "max_length": {"max_length": 64},
    "llm_pack_weight": {"llm_pack_weight": 0.5},
    "visual_pack_weight": {"visual_pack_weight": 0.5},
    "add_generation_prompt": {"add_generation_prompt": True},
    "enable_thinking": {"enable_thinking": False},
    "reasoning_effort": {"reasoning_effort": "low"},
}


@pytest.mark.parametrize("use_oss", [False, True])
def test_images_processed_incrementally_match_batched(tokenize_fn, tmp_path, monkeypatch, use_oss):
    from xtuner.v1.datasets.mllm_tokenize_fn import glm53_vl_tokenize_fn as module

    # Repeated shapes exercise grouping; the rotated long image checks EXIF and order.
    paths = []
    content = []
    rng = np.random.default_rng(42)
    for i, (width, height) in enumerate([(112, 84), (84, 560), (112, 84)]):
        path = tmp_path / f"image_{i}.png"
        image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
        exif = Image.Exif()
        if i == 1:
            exif[274] = 6
        image.save(path, exif=exif)
        image.close()
        paths.append(path)
        url = f"s3://bucket/{path.name}" if use_oss else str(path)
        content.append({"type": "image", "image": {"url": url, "image_wh": [width, height]}})
    content.append({"type": "text", "text": "Describe these images in order."})
    item = {"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": "Three images."}]}

    originals = [module.apply_exif_orientation(module.load_image(str(path))) for path in paths]
    expected = tokenize_fn.image_processor(images=originals, return_tensors="pt")
    for image in originals:
        image.close()
    del originals, image

    image_refs = []
    original_exif = module.apply_exif_orientation

    def track_image(image):
        image = original_exif(image)
        image_refs.append(weakref.ref(image))
        return image

    def check_previous_released():
        assert all(ref() is None for ref in image_refs), "Previous full-resolution image is still referenced"

    monkeypatch.setattr(module, "apply_exif_orientation", track_image)
    if use_oss:
        payloads = {f"s3://bucket/{path.name}": path.read_bytes() for path in paths}

        def get_image_bytes(url):
            check_previous_released()
            return payloads[url]

        tokenize_fn.oss_loader = Mock(client=Mock(get=Mock(side_effect=get_image_bytes)))
    else:
        original_load = module.load_image

        def load_image(path):
            check_previous_released()
            return original_load(path)

        monkeypatch.setattr(module, "load_image", load_image)

    tokenize_fn.state = "runtime"
    actual = tokenize_fn(item)
    assert len(image_refs) == 3
    check_previous_released()
    assert actual["num_imgs"] == [3]
    for key in ("pixel_values", "image_grid_thw"):
        assert torch.equal(actual[key], expected[key]), key
