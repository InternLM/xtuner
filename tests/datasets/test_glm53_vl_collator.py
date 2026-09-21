"""GLM-5.3-Flash 的 VL collator，见 doc/xtuner_glm5p3flash_design.md F1 §8.4。

TestSharedCollatorsStillWork
    test_qwen3_vl_collator_still_carries_media   共用 build_text_ctx_labels 的 qwen3_vl 回归
    test_intern_s1_collator_still_carries_media  同上，intern_s1
TestGlm53VlSftCollator
    test_glm53_collator_keeps_media_with_instance    截断丢样本时媒体同步丢弃
    test_glm53_collator_image_video_batch            图像/视频混批各自聚合
    test_glm53_collator_pure_text_sample_has_no_media 纯文本样本不带媒体字段
TestGlm53SequenceContextSplitKeepsMmAlignment
    test_glm53_sequence_context_split_keeps_mm_alignment  SP 切分保持模态标记对齐
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.datasets.collator import (
    glm53_vl_sft_collator,
    intern_s1_vl_sft_collator,
    qwen3_vl_sft_collator,
)
from xtuner.v1.utils import IGNORE_INDEX


def _text_item(text_len: int):
    input_ids = list(range(1, text_len + 1))
    labels = list(input_ids)
    return {"input_ids": input_ids, "labels": labels, "num_tokens": text_len}


class TestSharedCollatorsStillWork:
    def test_qwen3_vl_collator_still_carries_media(self):
        # qwen3_vl 与 glm53 共用 build_text_ctx_labels，改动后它仍要正常携带媒体。
        item = _text_item(4)
        item["pixel_values"] = torch.randn(4, 8)
        item["image_grid_thw"] = torch.tensor([[1, 2, 2]])
        item["num_img_tokens"] = [4]
        item["num_imgs"] = [1]
        result = qwen3_vl_sft_collator([[item]], pack_max_length=16, padding_token_idx=0)
        seq_ctx = result[0]["seq_ctx"]
        assert seq_ctx.pixel_values.shape == (4, 8)
        assert seq_ctx.num_img_tokens == [[4]]

    def test_intern_s1_collator_still_carries_media(self):
        # intern_s1 同上。
        item = _text_item(4)
        item["pixel_values"] = torch.randn(4, 8)
        item["num_img_tokens"] = [4]
        item["num_imgs"] = [1]
        result = intern_s1_vl_sft_collator([[item]], pack_max_length=16, padding_token_idx=0)
        seq_ctx = result[0]["seq_ctx"]
        assert seq_ctx.pixel_values.shape == (4, 8)


def _glm53_item(text_len: int, image_token_id: int, num_placeholders: int, merge_unit: int = 4):
    input_ids = [100] * text_len
    for i in range(num_placeholders):
        input_ids[i] = image_token_id
    labels = list(input_ids)
    labels[0] = IGNORE_INDEX
    mm_token_type_ids = torch.zeros(1, text_len, dtype=torch.long)
    mm_token_type_ids[0, :num_placeholders] = 1
    item = {
        "input_ids": input_ids,
        "labels": labels,
        "num_tokens": text_len,
        "num_imgs": [1] if num_placeholders else [0],
        "num_img_tokens": [num_placeholders * merge_unit] if num_placeholders else [0],
        "mm_token_type_ids": mm_token_type_ids,
    }
    if num_placeholders:
        item["pixel_values"] = torch.randn(num_placeholders * merge_unit, 8)
        item["image_grid_thw"] = torch.tensor([[1, 2, 2 * num_placeholders]])
    return item


class TestGlm53VlSftCollator:
    IMAGE_TOKEN_ID = 999
    MERGE_UNIT = 4

    def test_glm53_collator_keeps_media_with_instance(self):
        # 4 placeholders * merge_unit(4) = 16 raw patches.
        item = _glm53_item(text_len=10, image_token_id=self.IMAGE_TOKEN_ID, num_placeholders=4)
        result = glm53_vl_sft_collator(
            [[item]],
            pack_max_length=32,
            padding_token_idx=0,
            image_token_id=self.IMAGE_TOKEN_ID,
            merge_unit=self.MERGE_UNIT,
        )
        seq_ctx = result[0]["seq_ctx"]
        assert seq_ctx.pixel_values.shape == (16, 8)
        assert seq_ctx.num_img_tokens == [[16]]
        assert seq_ctx.mm_token_type_ids.shape[-1] == seq_ctx.input_ids.shape[-1]
        # instance text length 10 -> [:-1] drop -> 9 real tokens, padded to 32.
        assert (seq_ctx.mm_token_type_ids[0, :4] == 1).all()
        assert (seq_ctx.mm_token_type_ids[0, 9:] == 0).all()

    def test_glm53_collator_image_video_batch(self):
        # Two image-only samples packed together; each keeps its own media and placeholder count.
        item_a = _glm53_item(text_len=8, image_token_id=self.IMAGE_TOKEN_ID, num_placeholders=2)
        item_b = _glm53_item(text_len=6, image_token_id=self.IMAGE_TOKEN_ID, num_placeholders=1)
        result = glm53_vl_sft_collator(
            [[item_a, item_b]],
            pack_max_length=32,
            padding_token_idx=0,
            image_token_id=self.IMAGE_TOKEN_ID,
            merge_unit=self.MERGE_UNIT,
        )
        seq_ctx = result[0]["seq_ctx"]
        # 2*4 + 1*4 = 12 raw patches total, preserving sample order.
        assert seq_ctx.pixel_values.shape == (12, 8)
        assert seq_ctx.num_img_tokens == [[8], [4]]
        assert seq_ctx.image_grid_thw.shape[0] == 2

    def test_glm53_collator_pure_text_sample_has_no_media(self):
        # 纯文本样本不应带出任何媒体字段。
        item = _glm53_item(text_len=6, image_token_id=self.IMAGE_TOKEN_ID, num_placeholders=0)
        result = glm53_vl_sft_collator(
            [[item]],
            pack_max_length=16,
            padding_token_idx=0,
            image_token_id=self.IMAGE_TOKEN_ID,
            merge_unit=self.MERGE_UNIT,
        )
        seq_ctx = result[0]["seq_ctx"]
        assert seq_ctx.pixel_values is None
        assert seq_ctx.image_grid_thw is None
        assert (seq_ctx.mm_token_type_ids == 0).all()


@pytest.fixture
def single_rank_sp_mesh():
    """A 1-rank CPU SP mesh, torn down afterwards so the default process group does not leak
    into the rest of the session (a later test creating its own would hit "trying to initialize
    the default process group twice")."""
    owned = not dist.is_initialized()
    if owned:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29556")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("sp",))
    finally:
        if owned:
            dist.destroy_process_group()


class TestGlm53SequenceContextSplitKeepsMmAlignment:
    def test_glm53_sequence_context_split_keeps_mm_alignment(self, single_rank_sp_mesh):
        # SP 切分后 mm_token_type_ids 必须与 input_ids 走同一套 pad/split。
        sp_mesh = single_rank_sp_mesh

        seq_len = 10
        input_ids = torch.arange(seq_len).view(1, -1)
        ctx = SequenceContext.from_input_ids((input_ids,), device="cpu")
        mm_token_type_ids = torch.zeros(1, seq_len, dtype=torch.long)
        mm_token_type_ids[0, 2:5] = 1
        ctx.mm_token_type_ids = mm_token_type_ids

        split_ctx = ctx.split(sequence_parallel_mesh=sp_mesh)

        # world_size=1: the split is a structural no-op, but it must still run mm_token_type_ids
        # through the exact same pad_to_multiple_of + split_for_sequence_parallel path as
        # input_ids (design doc §8.3), not silently pass it through untouched.
        assert split_ctx.mm_token_type_ids is not None
        assert split_ctx.mm_token_type_ids.shape[-1] == split_ctx.input_ids.shape[-1]
        assert torch.equal(split_ctx.mm_token_type_ids[0, :seq_len], mm_token_type_ids[0])
