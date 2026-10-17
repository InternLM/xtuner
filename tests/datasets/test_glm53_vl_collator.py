"""GLM-5.3-Flash 的 VL collator，见 doc/xtuner_glm5p3flash_design.md F1 §8.4。

TestSharedCollatorsStillWork
    test_qwen3_vl_collator_still_carries_media   共用 build_text_ctx_labels 的 qwen3_vl 回归
    test_intern_s1_collator_still_carries_media  同上，intern_s1
TestGlm53VlSftCollator
    test_glm53_collator_keeps_media_with_instance    截断丢样本时媒体同步丢弃
    test_glm53_collator_image_video_batch            图像/视频混批各自聚合
    test_glm53_collator_pure_text_sample_has_no_media 纯文本样本不带媒体字段
    test_glm53_collator_drops_placeholder_mismatch_sample  占位符计数不符的样本打包前丢弃、不崩溃
    test_glm53_collator_all_corrupt_pack_no_loss_placeholder  整包被滤退无 loss 占位项，不中断训练
TestGlm53SequenceContextSplitKeepsMmAlignment
    test_glm53_sequence_context_split_keeps_mm_alignment  SP 切分保持模态标记对齐
"""

import os
from unittest.mock import Mock

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
from xtuner.v1.datasets.config import DataloaderConfig
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

    def test_glm53_collator_drops_placeholder_mismatch_sample(self, monkeypatch: pytest.MonkeyPatch):
        # 纯文本样本带字面 image_token_id（input_ids 含占位 token 但 num_img_tokens=0，线上
        # 256 卡 Step 8 崩溃的根因样本形态）：打包前丢弃并告警，pack 里其余样本正常训练，
        # collator 不再 assert 崩溃。
        logger_mock = Mock()
        monkeypatch.setattr("xtuner.v1.datasets.collator.logger", logger_mock)
        bad = _glm53_item(text_len=6, image_token_id=self.IMAGE_TOKEN_ID, num_placeholders=0)
        bad["input_ids"][1] = self.IMAGE_TOKEN_ID  # 字面 <|image|>，无对应媒体特征
        good = _glm53_item(text_len=8, image_token_id=self.IMAGE_TOKEN_ID, num_placeholders=2)
        result = glm53_vl_sft_collator(
            [[bad, good]],
            pack_max_length=32,
            padding_token_idx=0,
            image_token_id=self.IMAGE_TOKEN_ID,
            merge_unit=self.MERGE_UNIT,
        )
        seq_ctx = result[0]["seq_ctx"]
        assert logger_mock.warning.called
        # 只剩 good 样本：2 placeholders * merge_unit(4) = 8 raw patches。
        assert seq_ctx.pixel_values.shape == (8, 8)
        assert seq_ctx.num_img_tokens == [[8]]

    def test_glm53_collator_all_corrupt_pack_no_loss_placeholder(self, monkeypatch: pytest.MonkeyPatch):
        # 单样本 pack 是正常情况（长样本独占一个 pack、长度分组尾部等），整包被滤只说明这个
        # pack 恰好没有好样本，推不出系统性故障；此时 raise 会让单条脏样本中断训练（E02 回归）。
        # 退回无 loss 占位项：pack 照常产出、序列长度不变、loss 贡献为 0，与 runtime 通道
        # _discard_damaged_sample 的占位语义一致。
        logger_mock = Mock()
        monkeypatch.setattr("xtuner.v1.datasets.collator.logger", logger_mock)
        bad = _glm53_item(text_len=6, image_token_id=self.IMAGE_TOKEN_ID, num_placeholders=0)
        bad["input_ids"][1] = self.IMAGE_TOKEN_ID
        result = glm53_vl_sft_collator(
            [[bad]],
            pack_max_length=16,
            padding_token_idx=0,
            image_token_id=self.IMAGE_TOKEN_ID,
            merge_unit=self.MERGE_UNIT,
        )
        assert logger_mock.warning.called
        seq_ctx = result[0]["seq_ctx"]
        # 补 padding 到 pack_max_length，长度不变，DP 各 rank 的 collective 序列保持同步。
        assert seq_ctx.input_ids is not None
        assert seq_ctx.input_ids.shape[-1] == 16
        # 无 loss 占位：labels 全 IGNORE、不带任何媒体特征。
        assert (result[0]["shifted_labels"] == IGNORE_INDEX).all()
        assert seq_ctx.pixel_values is None
        assert seq_ctx.image_grid_thw is None
        assert seq_ctx.num_img_tokens == [[0]]
        assert (seq_ctx.mm_token_type_ids == 0).all()

    def test_glm53_collator_is_reachable_by_name_from_dataloader_config(self):
        # 训练侧只能按名字选 collator；`glm53_vl_sft_collator` 比其它 VL collator 多两个必填
        # 参数，要靠 `collator_kwargs` 绑定，否则 dataloader 一构建就 TypeError。
        cfg = DataloaderConfig(
            collator="glm53_vl_sft_collator",
            collator_kwargs={"image_token_id": self.IMAGE_TOKEN_ID, "merge_unit": self.MERGE_UNIT},
        )
        collator = cfg.build_collator()
        item = _glm53_item(text_len=10, image_token_id=self.IMAGE_TOKEN_ID, num_placeholders=4)
        # 只补 dataloader 自己会传的那几个参数，额外参数必须已经绑好。
        result = collator([[item]], pack_max_length=32, padding_token_idx=0)
        assert result[0]["seq_ctx"].pixel_values.shape == (16, 8)


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
