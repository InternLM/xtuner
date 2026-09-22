"""TrainingWorker 侧训练数据构造的 contract 测试（RolloutState → (seq_ctx, loss_ctx) → pack）。

重构后 token 级张量构造从 controller 移到 training worker，本文件只测两段公开纯逻辑，
不启动 Ray、模型或 rollout backend：

- ``_convert_rollout_items_to_train_items``：shift、advantage 张量化、seq_ctx 构造、
  distillation teacher targets 附加，并在 CPU 上构建 per-item loss_ctx
  （use_3d_position_ids 时给文本样本补 3D 轴）。
- ``_pack_train_items`` / ``_pack_one_batch``：按 controller 的 index-only 打包计划把
  (seq_ctx, loss_ctx) 拼成 ``pack_max_length`` 定长的 pack：loss_kwargs 张量沿序列维 cat、
  模板键（rollout_logprobs / teacher 字段）在空 pack 上以固定 padding 值填充，最后重建
  pack 级 loss_ctx（仍在 CPU，设备迁移与 sp 切分留给 ``_fit``），保证跨 rank collective
  形状一致。

当前测试点：
- shift 布局、advantage 与 shifted_labels 逐位置对齐（regression: advantage 曾比 input_ids
  长 1 导致 pack 后整体错位）。
- 混合批文本样本补 3D MRoPE 轴；VLM multimodal 字段与 3D position 原样保留。
- topk / sampled teacher targets 与 shifted 布局对齐（prompt 段补零行，语义洞保留行）。
- pack 内可选 loss 键必须 all-present 或 all-absent；空 pack 的模板键用 padding 值填充。
- padding 区取值表：labels=-100、advantages/rollout_logprobs=0、teacher_indices=-1、
  teacher 张量按 top_k 补零；3D position 补 arange、routed experts 补 dummy id。
"""

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import torch

from xtuner.v1.data_proto.rl_data import RolloutState, Status, TeacherTargets, write_train_meta
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.datasets.mllm_tokenize_fn.qwenvl_rope2d import get_rope_index_3
from xtuner.v1.rl.distillation import DistillationConfig, DistillationTrainerAdapter, RolloutTeacherConfig
from xtuner.v1.rl.loss import DistillationLossConfig, GRPOLossConfig
from xtuner.v1.rl.trainer.worker import TrainingWorker, TrainBatchAttr, get_train_seq_ctx


def _rollout_distillation_config(loss_config: DistillationLossConfig) -> DistillationConfig:
    return DistillationConfig(
        loss_config=loss_config,
        teachers=[RolloutTeacherConfig(name="teacher", endpoints=["http://teacher"])],
        data_source_teacher_map={"agent_math": "teacher"},
    )


def _make_worker(distillation_config=None) -> TrainingWorker:
    # 真实配置下 rollout teacher 依赖 loss_cfg 为 DistillationLossConfig，teacher 模板键才能进入 loss_ctx。
    loss_cfg = (
        distillation_config.loss_config
        if distillation_config is not None
        else GRPOLossConfig(policy_loss_cfg={"loss_type": "vanilla"})
    )
    worker = TrainingWorker.__new__(TrainingWorker)
    worker.config = SimpleNamespace(
        pack_max_length=16,
        loss_cfg=loss_cfg,
        distillation_config=distillation_config,
        model_cfg=SimpleNamespace(n_routed_experts=8),
    )
    worker._distillation = DistillationTrainerAdapter(distillation_config)
    return worker


def _make_state(
    *,
    uid: int = 1,
    prompt_ids: list[int] | None = None,
    response_ids: list[int] | None = None,
    logprobs: list[float] | None = None,
    supervised_mask: list[int] | None = None,
    input_ids: list[int] | None = None,
    labels: list[int] | None = None,
    position_ids: np.ndarray | None = None,
    mm_info: dict | None = None,
    routed_experts=None,
    teacher_tokens: list[int] | list[list[int]] | None = None,
    teacher_logprobs: list[float] | list[list[float]] | None = None,
) -> RolloutState:
    resolved_prompt_ids = prompt_ids if prompt_ids is not None else [10, 11, 12]
    resolved_response_ids = response_ids if response_ids is not None else [20, 21, 22]
    state = RolloutState(
        rollout_id=uid,
        group_id=1,
        message=[{"role": "user", "content": "prompt"}],
        prompt_ids=resolved_prompt_ids,
        response="response",
        response_ids=resolved_response_ids,
        logprobs=logprobs,
        reward={"score": 1.0},
        status=Status.COMPLETED,
        finish_reason="stop",
        routed_experts=routed_experts,
        position_ids=position_ids,
        mm_info=mm_info,
        extra_fields={"origin_data_source": "agent_math"} if teacher_tokens is not None else {},
        input_ids=input_ids,
        labels=labels,
        teacher_targets=(
            TeacherTargets(
                kind="topk" if isinstance(teacher_tokens[0], list) else "sampled",
                tokens=teacher_tokens,
                logprobs=teacher_logprobs,
            )
            if teacher_tokens is not None and teacher_logprobs is not None
            else None
        ),
    )
    if input_ids is None:
        # prompt+response 样本：模拟 loop 侧 canonicalize 后的最终形态（语义洞直接烙在 labels）。
        resp_ids = list(resolved_response_ids)
        state.input_ids = list(resolved_prompt_ids) + resp_ids
        if supervised_mask is None:
            state.labels = [-100] * len(resolved_prompt_ids) + resp_ids
        else:
            state.labels = [-100] * len(resolved_prompt_ids) + [
                resp_id if flag else -100 for resp_id, flag in zip(resp_ids, supervised_mask)
            ]
        if state.logprobs is not None:
            state.logprobs = [0.0] * len(resolved_prompt_ids) + list(state.logprobs)
    write_train_meta(state)
    return state


def _batch_attr(
    use_3d: bool = False,
    pack_loss_keys: list[str] | None = None,
    has_routed_experts: bool = False,
) -> TrainBatchAttr:
    return {
        "rollout_idx": 0,
        "use_3d_position_ids": use_3d,
        "pack_loss_keys": pack_loss_keys or [],
        "has_routed_experts": has_routed_experts,
    }


def _convert(worker: TrainingWorker, states: list[RolloutState], advantages: list[float], use_3d: bool = False):
    return worker._convert_rollout_items_to_train_items(states, advantages, _batch_attr(use_3d=use_3d))


class TestConvertRolloutItemsToTrainItems(unittest.TestCase):
    """RolloutState → (seq_ctx, loss_ctx)：shift、advantage 张量化、seq_ctx、teacher targets 与 CPU loss_ctx。"""

    def test_text_path_shifts_and_aligns_loss_inputs(self):
        # 文本主路径固定 token 布局：input_ids 去掉 response 最后一个 token，label/logprob 对齐预测位置。
        worker = _make_worker()
        routed_experts = np.array([[1, 2], [3, 4]])
        state = _make_state(
            response_ids=[20, 21, 22],
            logprobs=[0.1, 0.2, 0.3],
            supervised_mask=[1, 0, 1],
            routed_experts=routed_experts,
        )

        items = _convert(worker, [state], [1.5])

        self.assertEqual(len(items), 1)
        seq_ctx, loss_ctx = items[0]
        loss_kwargs = loss_ctx.loss_kwargs
        self.assertEqual(seq_ctx.input_ids.tolist(), [[10, 11, 12, 20, 21]])
        self.assertEqual(loss_kwargs.shifted_labels.tolist(), [[-100, -100, 20, -100, 22]])
        torch.testing.assert_close(
            loss_kwargs.rollout_logprobs,
            torch.tensor([[0.0, 0.0, 0.1, 0.2, 0.3]], dtype=torch.float32),
        )
        # advantage 与 shifted_labels 逐位置对齐: prompt 段为 0, mask=0 处为 0。
        torch.testing.assert_close(
            loss_kwargs.advantages,
            torch.tensor([[0.0, 0.0, 1.5, 0.0, 1.5]], dtype=torch.float32),
        )
        self.assertIs(seq_ctx.rollout_routed_experts, routed_experts)

    def test_mixed_batch_extends_text_positions_to_3d(self):
        # use_3d_position_ids=True 时 1D 文本样本沿三轴重复 arange，与 VLM 样本 pack 在一起。
        worker = _make_worker()
        reasoning = _make_state(
            uid=1,
            position_ids=np.arange(3, dtype=np.int64).reshape(1, 1, -1).repeat(3, axis=0),
        )
        agentic = _make_state(
            uid=2,
            input_ids=[30, 31, 40, 41, 42],
            labels=[-100, -100, 40, 41, 42],
            logprobs=[0.0, -0.1, -0.2, -0.3, -0.4],
        )

        items = _convert(worker, [reasoning, agentic], [0.5, 0.5], use_3d=True)

        self.assertEqual(tuple(items[0][0].position_ids.shape), (3, 1, 5))
        agentic_position_ids = items[1][0].position_ids
        self.assertEqual(tuple(agentic_position_ids.shape), (3, 1, 4))
        torch.testing.assert_close(
            agentic_position_ids,
            torch.arange(4, dtype=torch.long).reshape(1, 1, -1).expand(3, -1, -1),
        )

    def test_vlm_keeps_mm_fields_and_3d_positions(self):
        # VLM 样本保留 multimodal 字段；3D position_ids 按 response 段续写。
        worker = _make_worker()
        pixel_values = np.ones((1, 2, 3), dtype=np.float32)
        image_grid_thw = np.array([[1, 2, 3]], dtype=np.int32)
        position_ids = np.array([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=np.int64)
        state = _make_state(
            prompt_ids=[100, 101],
            response_ids=[102, 103],
            position_ids=position_ids,
            mm_info={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
        )

        items = _convert(worker, [state], [0.25], use_3d=True)

        seq_ctx = items[0][0]
        self.assertEqual(seq_ctx.input_ids.tolist(), [[100, 101, 102]])
        self.assertEqual(tuple(seq_ctx.position_ids.shape), (3, 1, 3))
        self.assertEqual(seq_ctx.position_ids.dtype, torch.long)
        self.assertIs(seq_ctx.pixel_values, pixel_values)
        self.assertEqual(seq_ctx.image_grid_thw.dtype, torch.long)
        self.assertEqual(seq_ctx.image_grid_thw.tolist(), [[1, 2, 3]])

    def test_get_train_seq_ctx_mrope_continues_from_global_amax(self):
        """RL 只拿 prompt 的 3D position，续写 response 后须与 SFT get_rope_index_3 一致。

        Fixture：prompt 以 image tokens 结尾（grid 1x4x4, merge=2 → 2x2），使 per-axis max
        与 global amax 分叉。
        """
        image_token_id = 151655
        vision_start_token_id = 151652
        prompt_ids = [10, vision_start_token_id, image_token_id, image_token_id, image_token_id, image_token_id]
        response_ids = [20, 21, 22]
        full_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)

        sft_position_ids = get_rope_index_3(
            full_ids,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=2,
        )
        prompt_position_ids = get_rope_index_3(
            torch.tensor([prompt_ids], dtype=torch.long),
            image_grid_thw=image_grid_thw,
            spatial_merge_size=2,
        )

        seq_ctx = get_train_seq_ctx(cast(torch.LongTensor, full_ids), prompt_position_ids.numpy())
        rl_position_ids = seq_ctx.position_ids
        assert rl_position_ids is not None
        self.assertEqual(tuple(rl_position_ids.shape), tuple(sft_position_ids.shape))
        torch.testing.assert_close(rl_position_ids, sft_position_ids)

    def test_topk_teacher_targets_align_shifted_layout(self):
        # topk targets 补齐 prompt 段零行后与 shifted_labels 逐行对齐，并可直接构建 loss。
        loss_config = DistillationLossConfig(
            policy_loss_cfg={
                "loss_type": "vanilla",
                "cliprange_low": 0.2,
                "cliprange_high": 0.2,
            },
            loss_mode="reverse",
            use_policy_gradient=False,
            top_k=2,
        )
        worker = _make_worker(_rollout_distillation_config(loss_config))
        state = _make_state(
            input_ids=[10, 11, 20, 21, 22],
            labels=[-100, -100, 20, -100, 22],
            logprobs=[0.0, -0.1, -0.2, -0.3, -0.4],
            teacher_tokens=[[100, 101], [102, 103], [104, 105]],
            teacher_logprobs=[[-0.5, -0.6], [-0.7, -0.8], [-0.9, -1.0]],
        )

        items = _convert(worker, [state], [0.0])

        self.assertEqual(len(items), 1)
        loss_kwargs = items[0][1].loss_kwargs
        self.assertEqual(loss_kwargs.shifted_labels.tolist(), [[-100, 20, -100, 22]])
        self.assertEqual(
            loss_kwargs.target_token_ids.tolist(),
            [[[0, 0], [100, 101], [102, 103], [104, 105]]],
        )
        torch.testing.assert_close(
            loss_kwargs.teacher_logprobs,
            torch.tensor([[[0.0, 0.0], [-0.5, -0.6], [-0.7, -0.8], [-0.9, -1.0]]], dtype=torch.float32),
        )
        with patch("xtuner.v1.rl.loss.distillation_loss.DEVICE", "cpu"):
            loss_ctx = loss_config.build(
                {
                    "shifted_labels": loss_kwargs.shifted_labels,
                    "advantages": loss_kwargs.advantages,
                    "old_logprobs": torch.zeros_like(loss_kwargs.shifted_labels, dtype=torch.float32),
                    "teacher_logprobs": loss_kwargs.teacher_logprobs,
                    "target_token_ids": loss_kwargs.target_token_ids,
                }
            )
        assert loss_ctx is not None
        type(loss_ctx).build_batches([loss_ctx])
        loss, _ = loss_ctx.loss_fn(
            hidden_states=torch.randn(1, 4, 8),
            head_weight=torch.randn(128, 8),
            head_bias=None,
            loss_kwargs=loss_ctx.loss_kwargs,
        )
        self.assertTrue(torch.isfinite(loss))

    def test_topk_targets_include_masked_response_rows(self):
        # labels=-100 的 response 行也保留 teacher 行（loss 侧自行忽略），只在前面补零行。
        loss_config = DistillationLossConfig(
            policy_loss_cfg={"loss_type": "vanilla"},
            loss_mode="forward_kl_topk",
            use_policy_gradient=False,
            top_k=2,
        )
        worker = _make_worker(_rollout_distillation_config(loss_config))
        state = _make_state(
            prompt_ids=[10, 11, 12],
            response_ids=[20, 21, 22],
            supervised_mask=[0, 1, 1],
            teacher_tokens=[[102, 103], [104, 105]],
            teacher_logprobs=[[-0.7, -0.8], [-0.9, -1.0]],
        )

        items = _convert(worker, [state], [0.0])

        loss_kwargs = items[0][1].loss_kwargs
        self.assertEqual(loss_kwargs.shifted_labels.tolist(), [[-100, -100, -100, 21, 22]])
        self.assertEqual(
            loss_kwargs.target_token_ids.tolist(),
            [[[0, 0], [0, 0], [0, 0], [102, 103], [104, 105]]],
        )
        torch.testing.assert_close(
            loss_kwargs.teacher_logprobs,
            torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [-0.7, -0.8], [-0.9, -1.0]]], dtype=torch.float32),
        )

    def test_sampled_targets_have_no_token_ids(self):
        # sampled（k1）目标只有逐位置 teacher logprobs，不产出 target_token_ids 键。
        loss_config = DistillationLossConfig(
            policy_loss_cfg={"loss_type": "vanilla"},
            loss_mode="k1",
            use_policy_gradient=True,
        )
        worker = _make_worker(_rollout_distillation_config(loss_config))
        plain_state = _make_state(
            uid=1,
            prompt_ids=[10, 11, 12],
            response_ids=[20, 21, 22],
            supervised_mask=[0, 1, 1],
            teacher_tokens=[21, 22],
            teacher_logprobs=[-0.7, -0.9],
        )
        agentic_state = _make_state(
            uid=2,
            input_ids=[30, 31, 40, 41, 42],
            labels=[-100, -100, 40, -100, 42],
            logprobs=[0.0, -0.1, -0.2, -0.3, -0.4],
            teacher_tokens=[40, 41, 42],
            teacher_logprobs=[-1.1, -1.2, -1.3],
        )

        items = _convert(worker, [plain_state, agentic_state], [0.0, 0.0])

        self.assertEqual(len(items), 2)
        torch.testing.assert_close(
            items[0][1].loss_kwargs.teacher_logprobs,
            torch.tensor([[0.0, 0.0, 0.0, -0.7, -0.9]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            items[1][1].loss_kwargs.teacher_logprobs,
            torch.tensor([[0.0, -1.1, -1.2, -1.3]], dtype=torch.float32),
        )
        self.assertIsNone(items[0][1].loss_kwargs.target_token_ids)
        self.assertIsNone(items[1][1].loss_kwargs.target_token_ids)


class TestPackTrainItems(unittest.TestCase):
    """(seq_ctx, loss_ctx) 列表 → 定长 pack：拼接、padding 值表、模板键与 pack 级 loss_ctx 重建。"""

    def _pack(self, worker, items, plan, use_3d=False, pack_loss_keys=None, has_routed_experts=False):
        return worker._pack_train_items(items, plan, _batch_attr(use_3d, pack_loss_keys, has_routed_experts))

    def test_packs_concatenate_along_sequence(self):
        # 两个样本按 plan 顺序拼进一个 pack，padding 区补满 pack_max_length。
        worker = _make_worker()
        first = _make_state(uid=1, response_ids=[20, 21], logprobs=[0.1, 0.2])
        second = _make_state(uid=2, prompt_ids=[30, 31], response_ids=[40], logprobs=[-0.3])
        items = _convert(worker, [first, second], [1.0, 2.0])

        step_batches = self._pack(worker, items, [[[0, 1]]])

        self.assertEqual(len(step_batches), 1)
        seq_ctx, loss_ctx = step_batches[0][0]
        seq_len = seq_ctx.input_ids.shape[1]
        self.assertEqual(seq_len, worker.config.pack_max_length)
        # first: prompt 3 + response 2 → 4 个训练位; second: 2 个训练位; padding 10。
        self.assertEqual(seq_ctx.num_padding, 16 - 6)
        loss_kwargs = loss_ctx.loss_kwargs
        # pack 阶段产物保持 CPU，设备迁移由 _fit 统一执行。
        self.assertEqual(loss_kwargs.shifted_labels.device.type, "cpu")
        # 真实段逐位置保序；padding 区 labels=-100、advantages=0。
        self.assertEqual(loss_kwargs.shifted_labels[0, :4].tolist(), [-100, -100, 20, 21])
        self.assertEqual(loss_kwargs.shifted_labels[0, 4:6].tolist(), [-100, 40])
        torch.testing.assert_close(
            loss_kwargs.advantages[0, :6],
            torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 2.0]),
        )
        pad = loss_kwargs.shifted_labels[0, 6:]
        self.assertTrue(torch.all(pad == -100))
        self.assertTrue(torch.all(loss_kwargs.advantages[0, 6:] == 0))

    def test_single_item_without_padding_keeps_seq_ctx(self):
        # 恰好一个样本且无 padding 时不经 cat，seq_ctx 原样保留（含 multimodal 字段）。
        worker = _make_worker()
        state = _make_state(
            uid=1,
            prompt_ids=[10, 11, 12, 10, 11, 12, 10, 11, 12, 10, 11, 12, 10, 11, 12, 10],
            response_ids=[20],
        )
        items = _convert(worker, [state], [1.0])
        self.assertEqual(items[0]["seq_ctx"].input_ids.shape[1], worker.config.pack_max_length)

        step_batches = self._pack(worker, items, [[[0]]])

        seq_ctx, _ = step_batches[0][0]
        self.assertIs(seq_ctx, items[0]["seq_ctx"])
        self.assertEqual(seq_ctx.num_padding, 0)

    def test_empty_pack_materializes_all_padding(self):
        # 调度占位空 pack：整包都是 padding（全零 input_ids + labels=-100 + advantages=0）。
        worker = _make_worker()

        step_batches = self._pack(worker, [], [[]])

        seq_ctx, loss_ctx = step_batches[0][0]
        seq_len = seq_ctx.input_ids.shape[1]
        self.assertEqual(seq_len, worker.config.pack_max_length)
        self.assertEqual(seq_ctx.num_padding, 16)
        self.assertTrue(torch.all(seq_ctx.input_ids == 0))
        loss_kwargs = loss_ctx.loss_kwargs
        self.assertTrue(torch.all(loss_kwargs.shifted_labels == -100))
        self.assertTrue(torch.all(loss_kwargs.advantages == 0))

    def test_rollout_logprobs_template_key_on_empty_pack(self):
        # pack_loss_keys 声明的批级模板键在空 pack 上也要物化，保证跨 rank collective 形状一致。
        worker = _make_worker()

        step_batches = self._pack(worker, [], [[]], pack_loss_keys=["rollout_logprobs"])

        _, loss_ctx = step_batches[0][0]
        rollout_logprobs = loss_ctx.loss_kwargs.rollout_logprobs
        assert rollout_logprobs is not None
        torch.testing.assert_close(
            rollout_logprobs,
            torch.zeros(1, worker.config.pack_max_length, dtype=torch.float32),
        )

    def test_teacher_template_keys_padded_on_empty_pack(self):
        # 配置 distillation 后空 pack 也要带 teacher 模板键（top_k 形状补零），供全 pad rank 对齐。
        loss_config = DistillationLossConfig(
            policy_loss_cfg={"loss_type": "vanilla"},
            loss_mode="forward_kl_topk",
            use_policy_gradient=False,
            top_k=2,
        )
        worker = _make_worker(_rollout_distillation_config(loss_config))

        step_batches = self._pack(worker, [], [[]])

        _, loss_ctx = step_batches[0][0]
        loss_kwargs = loss_ctx.loss_kwargs
        torch.testing.assert_close(
            loss_kwargs.teacher_logprobs,
            torch.zeros(1, worker.config.pack_max_length, 2, dtype=torch.float32),
        )
        torch.testing.assert_close(
            loss_kwargs.target_token_ids,
            torch.zeros(1, worker.config.pack_max_length, 2, dtype=torch.long),
        )

    def test_mixed_presence_within_pack_raises(self):
        # 同一 pack 内可选 loss 键必须 all-present 或 all-absent，混出即 fail fast。
        worker = _make_worker()
        loss_cfg = GRPOLossConfig(policy_loss_cfg={"loss_type": "vanilla"})
        seq_ctx_a = SequenceContext.from_input_ids((torch.tensor([[1, 2]]),), device="cpu")
        seq_ctx_b = SequenceContext.from_input_ids((torch.tensor([[3, 4]]),), device="cpu")
        items = [
            (
                seq_ctx_a,
                loss_cfg.build(
                    {
                        "shifted_labels": torch.tensor([[-100, 2]]),
                        "advantages": torch.zeros(1, 2),
                        "rollout_logprobs": torch.zeros(1, 2),
                    },
                    device="cpu",
                ),
            ),
            (
                seq_ctx_b,
                loss_cfg.build(
                    {
                        "shifted_labels": torch.tensor([[3, 4]]),
                        "advantages": torch.zeros(1, 2),
                    },
                    device="cpu",
                ),
            ),
        ]

        with self.assertRaisesRegex(ValueError, "all-present or all-absent"):
            self._pack(worker, items, [[[0, 1]]], pack_loss_keys=["rollout_logprobs"])

    def test_oversize_pack_raises(self):
        # pack 内真实 token 数超过 pack_max_length 时 fail fast（controller 侧 packer 同样拦截）。
        worker = _make_worker()
        long_ids = list(range(100, 120))
        state = _make_state(
            uid=1,
            input_ids=long_ids,
            labels=[-100] * 19 + [42],
            logprobs=[0.0] * 20,
        )
        items = _convert(worker, [state], [1.0])

        with self.assertRaises(AssertionError):
            self._pack(worker, items, [[[0]]])

    def test_padding_seq_ctx_uses_3d_positions_when_configured(self):
        # use_3d_position_ids=True 时 padding 区 position_ids 为三轴 arange。
        worker = _make_worker()

        step_batches = self._pack(worker, [], [[]], use_3d=True)

        position_ids = step_batches[0][0][0].position_ids
        self.assertEqual(tuple(position_ids.shape), (3, 1, worker.config.pack_max_length))
        torch.testing.assert_close(
            position_ids,
            torch.arange(worker.config.pack_max_length, dtype=torch.long).reshape(1, 1, -1).expand(3, -1, -1),
        )

    def test_padding_seq_ctx_gets_dummy_routed_experts(self):
        # has_routed_experts=True 时 padding 区补 dummy expert id，形状与真实 rollout_routed_experts 对齐。
        worker = _make_worker()

        step_batches = self._pack(worker, [], [[]], has_routed_experts=True)

        routed_experts = step_batches[0][0][0].rollout_routed_experts
        self.assertEqual(tuple(routed_experts.shape), (worker.config.pack_max_length, 1, 1))
        self.assertTrue(torch.all(routed_experts >= 0))
        self.assertTrue(torch.all(routed_experts < worker.config.model_cfg.n_routed_experts))

    def test_configured_teacher_loss_keys(self):
        # 模板键由 distillation 配置推导：rollout topk → logprobs+token_ids；sampled → 仅 logprobs；
        # train teachers → teacher_indices；未配置 → 空。
        topk_worker = _make_worker(
            _rollout_distillation_config(
                DistillationLossConfig(
                    policy_loss_cfg={"loss_type": "vanilla"},
                    loss_mode="forward_kl_topk",
                    use_policy_gradient=False,
                    top_k=2,
                )
            )
        )
        self.assertEqual(topk_worker._configured_teacher_loss_keys(), ["teacher_logprobs", "target_token_ids"])

        sampled_worker = _make_worker(
            _rollout_distillation_config(
                DistillationLossConfig(
                    policy_loss_cfg={"loss_type": "vanilla"},
                    loss_mode="k1",
                    use_policy_gradient=True,
                )
            )
        )
        self.assertEqual(sampled_worker._configured_teacher_loss_keys(), ["teacher_logprobs"])

        train_teacher_config = SimpleNamespace(
            rollout_teachers=[],
            train_teachers=[object()],
            loss_config=SimpleNamespace(uses_sampled_token_targets=True),
        )
        train_teacher_worker = _make_worker()
        train_teacher_worker.config.distillation_config = train_teacher_config
        self.assertEqual(train_teacher_worker._configured_teacher_loss_keys(), ["teacher_indices"])

        self.assertEqual(_make_worker()._configured_teacher_loss_keys(), [])


if __name__ == "__main__":
    unittest.main()
