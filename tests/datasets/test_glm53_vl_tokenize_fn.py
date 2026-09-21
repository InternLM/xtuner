"""GLM-5.3-Flash 的 VL TokenizeFn（cache/runtime 双路径），见设计文档 F1。

TestGlm53TokenizeCacheRuntimeParity
    test_glm53_tokenize_cache_runtime_image_parity  图像：cache 预测与 runtime 实算一致
    test_glm53_tokenize_cache_runtime_video_parity  视频：同上
TestGlm53VisualPlaceholderAndTokenSemantics
    test_glm53_visual_placeholder_count             占位符数量等于 patch 数 / merge_unit
    test_glm53_num_img_tokens_is_per_vit_sequence   num_img_tokens 按 ViT 序列计数
    test_glm53_mm_token_type_ids_alignment          mm_token_type_ids 与 input_ids 等长对齐
TestGlm53MmTokenTypeIdsMatchesProcessorGolden
    test_image_only                                 图像模态标记与 HF processor 一致
    test_video_only                                 视频模态标记与 HF processor 一致
TestGlm53MixedMediaAndTruncation
    test_glm53_mixed_media_fails_fast               图文混合视频未支持，必须快速失败
    test_glm53_visual_truncation_is_dropped         截断会切断视觉跨度时丢弃该样本
TestGlm53CacheInvalidation
    test_glm53_cache_invalidates_on_processor_or_pack_change  processor/pack 变化使缓存失效
"""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from transformers import AutoProcessor, AutoTokenizer
from transformers.video_utils import VideoMetadata
from xtuner.v1.datasets.mllm_tokenize_fn import Glm53VLTokenizeFnConfig


GLM_5_3_FLASH_PATH = os.environ.get(
    "GLM_5_3_FLASH_PATH", "/mnt/shared-storage-user/zhaopenghao/model/GLM-5.3-Flash-25B"
)


@pytest.fixture(scope="module")
def ckpt_path():
    if not os.path.isdir(GLM_5_3_FLASH_PATH):
        pytest.skip(f"GLM_5_3_FLASH_PATH not found: {GLM_5_3_FLASH_PATH}")
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


def _video_item(video_dir: str, num_frames: int = 6, fps: float = 2.0, width: int = 224, height: int = 224):
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


class TestGlm53MixedMediaAndTruncation:
    def test_glm53_mixed_media_fails_fast(self, tokenize_fn, image_fixture, video_fixture):
        item = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": {"url": image_fixture, "image_wh": [448, 448]}},
                        {
                            "type": "video",
                            "video": {
                                "url": video_fixture,
                                "image_wh": [224, 224],
                                "origin_video_length": 6,
                                "origin_fps": 2.0,
                            },
                        },
                    ],
                }
            ]
        }
        tokenize_fn.state = "runtime"
        with pytest.raises(NotImplementedError):
            tokenize_fn(item)

    def test_glm53_visual_truncation_is_dropped(self, ckpt_path, tokenizer, image_fixture):
        # max_length shorter than the un-truncated sample cuts into the 256-token image span;
        # the resulting placeholder-count mismatch must raise, not silently train on half a span.
        short_fn = Glm53VLTokenizeFnConfig(processor_path=ckpt_path, max_length=20).build(tokenizer, anno_name="test")
        short_fn.state = "runtime"
        with pytest.raises(AssertionError):
            short_fn(_image_item(image_fixture))


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
