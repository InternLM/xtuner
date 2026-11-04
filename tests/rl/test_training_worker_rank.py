"""角色拆分后训练数据构造的 contract 测试（RolloutState → TrainSample → (seq_ctx, loss_ctx) pack）。

token 级张量构造由宿主 ``TrainingWorker`` 与角色 worker 分两段完成，本文件只测这两段公开
纯逻辑，不启动 Ray actor、模型或 rollout backend：

- ``TrainingWorker._get_train_sample`` + ``_add_sample_fields``：shift、seq_ctx 构造、
  distillation teacher targets 附加（经 ``ActorWorker.get_rollout_teacher_targets``）、
  advantage 经样本缓存写入（``TrainSample.advantages``；use_3d_position_ids 时给文本样本
  补 3D 轴）。loss context 构造被推迟到 pack 阶段。
- ``TrainingWorker.pack_seq_ctx`` + ``ActorWorker.pack_loss_ctx``（内部走
  ``_pack_loss_fields`` / ``RoleWorker._pack_loss_rows``）：按 controller 的 index-only
  打包计划把样本拼成 ``pack_max_length`` 定长的 pack：seq_ctx 与 loss_kwargs 张量沿序列维
  cat、模板键（rollout_logprobs / teacher 字段）在空 pack 上以固定 padding 值填充，最后
  重建 pack 级 loss_ctx（仍在 CPU），保证跨 rank collective 形状一致。
- ``resolve_routed_experts``（role_worker 自由函数）：把 rollout 侧 routed-experts
  object ref 物化为 pack 级 CPU 张量（segmented 语义保留 ``[1:]``）。

当前测试点：
- shift 布局、advantage 与 shifted_labels 逐位置对齐（regression: advantage 曾比 input_ids
  长 1 导致 pack 后整体错位）。
- 混合批文本样本补 3D MRoPE 轴；VLM multimodal 字段与 3D position 原样保留。
- topk / sampled teacher targets 与 shifted 布局对齐（prompt 段补零行，语义洞保留行）。
- pack 模板要求的 teacher loss 键必须在所有真实样本中存在；空 pack 的模板键用 padding 值填充。
- padding 区取值表：labels=-100、advantages/rollout_logprobs=0、teacher_indices=-1、
  teacher 张量按 top_k 补零；3D position 补 arange、routed experts 补 dummy id。
"""

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import ray
import torch

from xtuner.v1.data_proto.rl_data import RolloutState, Status, TeacherTargets
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.datasets.mllm_tokenize_fn.qwenvl_rope2d import get_rope_index_3
from xtuner.v1.rl.agent_loop.agent_loop import AgentLoop
from xtuner.v1.rl.distillation import DistillationConfig, DistillationTrainerAdapter, RolloutTeacherConfig
from xtuner.v1.rl.loss import DistillationLossConfig, GRPOLossConfig
from xtuner.v1.rl.trainer.actor_worker import ActorWorker
from xtuner.v1.rl.trainer.pack import RLDataPacker
from xtuner.v1.rl.trainer.role_worker import resolve_routed_experts
from xtuner.v1.rl.trainer.worker import TrainBatchAttr, TrainingWorker, TrainSample, get_train_seq_ctx


def _rollout_distillation_config(loss_config: DistillationLossConfig) -> DistillationConfig:
    return DistillationConfig(
        loss_config=loss_config,
        teachers=[RolloutTeacherConfig(name="teacher", endpoints=["http://teacher"])],
        data_source_teacher_map={"agent_math": "teacher"},
    )


def _train_teacher_config() -> SimpleNamespace:
    # 只配置 train teachers（最小 distillation 形状），用于 teacher_indices 模板键路径。
    return SimpleNamespace(
        rollout_teachers=[],
        train_teachers=[object()],
        loss_config=SimpleNamespace(uses_sampled_token_targets=True),
    )


def _make_worker(distillation_config=None, model_cfg=None) -> TrainingWorker:
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
        model_cfg=model_cfg or SimpleNamespace(n_routed_experts=8),
    )
    worker.sp_mesh = SimpleNamespace(size=lambda: 1)
    worker._cached_samples = None
    worker._cached_pack_plan = None
    worker._cached_batch_attr = None
    actor = ActorWorker.__new__(ActorWorker)
    actor.config = worker.config
    actor._distillation = DistillationTrainerAdapter(distillation_config)
    actor._staged = {}
    actor.mtp_config = None
    actor.host = worker
    worker.actor = actor
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
    AgentLoop._write_train_meta(None, state)
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


def _convert(
    worker: TrainingWorker,
    states: list[RolloutState],
    advantages: list[float] | None = None,
    use_3d: bool = False,
) -> list[TrainSample]:
    # 宿主侧转换 + advantage 写入缓存：转换阶段不触 loss context。
    samples = [worker._get_train_sample(state, use_3d) for state in states]
    worker._cached_samples = samples
    if advantages is not None:
        worker._add_sample_fields("advantages", advantages)
    return samples


def _sample_fields(sample: TrainSample) -> dict[str, torch.Tensor]:
    # 一个样本的全部 CPU loss 输入行（键名与 packed loss_kwargs 一致）。
    fields = {"shifted_labels": sample.shifted_labels}
    if sample.advantages is not None:
        fields["advantages"] = sample.advantages
    if sample.rollout_logprobs is not None:
        fields["rollout_logprobs"] = sample.rollout_logprobs
    fields.update(sample.teacher_targets)
    return fields


def _pack(
    worker: TrainingWorker,
    samples: list[TrainSample],
    plan,
    use_3d: bool = False,
    pack_loss_keys: list[str] | None = None,
    has_routed_experts: bool = False,
) -> list[list[tuple[SequenceContext, object]]]:
    # 复刻 actor fit 的 pack 路径：宿主拼 seq_ctx，角色拼 loss_ctx；step 布局与 plan 一致。
    worker._cached_samples = samples
    worker._cached_pack_plan = plan
    worker._cached_batch_attr = _batch_attr(use_3d, pack_loss_keys, has_routed_experts)
    steps = []
    with patch("xtuner.v1.rl.trainer.actor_worker.DEVICE", "cpu"):
        for step in plan:
            packed = []
            for pack in step:
                seq_ctx = worker.pack_seq_ctx(pack)
                loss_dict = worker.actor.pack_loss_ctx(pack, seq_ctx)
                packed.append((seq_ctx, loss_dict["lm"]))
            steps.append(packed)
    return steps


class TestConvertRolloutItemsToTrainSamples(unittest.TestCase):
    """RolloutState → TrainSample：shift、advantage 写入、seq_ctx、teacher targets；loss ctx 延迟构造。"""

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

        samples = _convert(worker, [state], [1.5])

        self.assertEqual(len(samples), 1)
        seq_ctx = samples[0].seq_ctx
        self.assertEqual(seq_ctx.input_ids.tolist(), [[10, 11, 12, 20, 21]])
        self.assertEqual(samples[0].shifted_labels.tolist(), [[-100, -100, 20, -100, 22]])
        torch.testing.assert_close(
            samples[0].rollout_logprobs,
            torch.tensor([[0.0, 0.0, 0.1, 0.2, 0.3]], dtype=torch.float32),
        )
        # advantage 与 shifted_labels 逐位置对齐: prompt 段为 0, mask=0 处为 0。
        torch.testing.assert_close(
            samples[0].advantages,
            torch.tensor([[0.0, 0.0, 1.5, 0.0, 1.5]], dtype=torch.float32),
        )
        self.assertIs(seq_ctx.rollout_routed_experts, routed_experts)

    def test_conversion_defers_loss_context_construction_until_packing(self):
        # 转换 + advantage 写入阶段不 build loss context；pack 阶段才构建。
        worker = _make_worker()
        with patch.object(type(worker.config.loss_cfg), "build", side_effect=AssertionError("premature loss context")):
            samples = _convert(worker, [_make_state()], [1.0])
        self.assertIsNotNone(samples[0].advantages)
        self.assertIsNotNone(samples[0].seq_ctx.input_ids)
        packed = _pack(worker, samples, [[[0]]])
        self.assertEqual(packed[0][0][1].loss_kwargs.shifted_labels.shape[1], worker.config.pack_max_length)

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

        samples = _convert(worker, [reasoning, agentic], [0.5, 0.5], use_3d=True)

        self.assertEqual(tuple(samples[0].seq_ctx.position_ids.shape), (3, 1, 5))
        agentic_position_ids = samples[1].seq_ctx.position_ids
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

        samples = _convert(worker, [state], [0.25], use_3d=True)

        seq_ctx = samples[0].seq_ctx
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

        samples = _convert(worker, [state], [0.0])

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample.shifted_labels.tolist(), [[-100, 20, -100, 22]])
        self.assertEqual(
            sample.teacher_targets["target_token_ids"].tolist(),
            [[[0, 0], [100, 101], [102, 103], [104, 105]]],
        )
        torch.testing.assert_close(
            sample.teacher_targets["teacher_logprobs"],
            torch.tensor([[[0.0, 0.0], [-0.5, -0.6], [-0.7, -0.8], [-0.9, -1.0]]], dtype=torch.float32),
        )
        with patch("xtuner.v1.rl.loss.distillation_loss.DEVICE", "cpu"):
            loss_ctx = loss_config.build(
                {
                    "shifted_labels": sample.shifted_labels,
                    "advantages": sample.advantages,
                    "old_logprobs": torch.zeros_like(sample.shifted_labels, dtype=torch.float32),
                    "teacher_logprobs": sample.teacher_targets["teacher_logprobs"],
                    "target_token_ids": sample.teacher_targets["target_token_ids"],
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

        samples = _convert(worker, [state], [0.0])

        sample = samples[0]
        self.assertEqual(sample.shifted_labels.tolist(), [[-100, -100, -100, 21, 22]])
        self.assertEqual(
            sample.teacher_targets["target_token_ids"].tolist(),
            [[[0, 0], [0, 0], [0, 0], [102, 103], [104, 105]]],
        )
        torch.testing.assert_close(
            sample.teacher_targets["teacher_logprobs"],
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

        samples = _convert(worker, [plain_state, agentic_state], [0.0, 0.0])

        self.assertEqual(len(samples), 2)
        torch.testing.assert_close(
            samples[0].teacher_targets.get("teacher_logprobs"),
            torch.tensor([[0.0, 0.0, 0.0, -0.7, -0.9]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            samples[1].teacher_targets.get("teacher_logprobs"),
            torch.tensor([[0.0, -1.1, -1.2, -1.3]], dtype=torch.float32),
        )
        self.assertIsNone(samples[0].teacher_targets.get("target_token_ids"))
        self.assertIsNone(samples[1].teacher_targets.get("target_token_ids"))


class TestPackTrainItems(unittest.TestCase):
    """TrainSample 列表 → 定长 pack：拼接、padding 值表、模板键与 pack 级 loss_ctx 重建。"""

    def test_packs_concatenate_along_sequence(self):
        # 两个样本按 plan 顺序拼进一个 pack，padding 区补满 pack_max_length。
        worker = _make_worker()
        first = _make_state(uid=1, response_ids=[20, 21], logprobs=[0.1, 0.2])
        second = _make_state(uid=2, prompt_ids=[30, 31], response_ids=[40], logprobs=[-0.3])
        samples = _convert(worker, [first, second], [1.0, 2.0])

        step_batches = _pack(worker, samples, [[[0, 1]]])

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

    def test_pack_fields_round_trip_to_original_samples(self):
        # 各字段共用同一份真实 pack plan；按样本边界拆回后应逐元素一致。
        # 长度和字段值各不相同，避免相同值掩盖样本重排、漏样本或字段错位。
        worker = _make_worker(
            _rollout_distillation_config(
                DistillationLossConfig(
                    policy_loss_cfg={"loss_type": "vanilla"},
                    loss_mode="forward_kl_topk",
                    use_policy_gradient=False,
                    top_k=2,
                )
            )
        )
        worker.config.pack_max_length = 10
        lengths = [6, 7, 4, 10, 3, 8, 5]
        states = []
        for index, length in enumerate(lengths):
            base = (index + 1) * 100
            response = list(range(base + 2, base + length + 1))
            states.append(
                _make_state(
                    uid=index + 1,
                    prompt_ids=[base, base + 1],
                    response_ids=response,
                    supervised_mask=[int(position % 3 != 1) for position in range(len(response))],
                    logprobs=[-(base + position) / 1000 for position in range(len(response))],
                    teacher_tokens=[[token, token + 1000] for token in response],
                    teacher_logprobs=[[-token / 1000, -token / 2000] for token in response],
                )
            )
        samples = _convert(worker, states, [float(index + 1) for index in range(len(states))])
        originals = [
            {
                "input_ids": sample.seq_ctx.input_ids.clone(),
                "position_ids": sample.seq_ctx.position_ids.clone(),
                **{key: value.clone() for key, value in _sample_fields(sample).items()},
            }
            for sample in samples
        ]
        for strategy in ("legacy", "greedy", "balance", "native"):
            with self.subTest(strategy=strategy):
                with patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", True):
                    plans, _ = RLDataPacker(10, 3, 1, 2, pack_strategy=strategy, pack_seed=7).pack(lengths)
                restored = {}
                for plan in plans:
                    batches = _pack(worker, samples, plan, pack_loss_keys=["rollout_logprobs"])
                    self.assertEqual(len(batches), len(plan))
                    for step_plan, step_batches in zip(plan, batches):
                        self.assertEqual(len(step_batches), len(step_plan))
                        for indices, (seq, loss) in zip(step_plan, step_batches):
                            fields = {
                                "input_ids": seq.input_ids,
                                "position_ids": seq.position_ids,
                                **{
                                    key: getattr(loss.loss_kwargs, key)
                                    for key in originals[0]
                                    if key not in ("input_ids", "position_ids")
                                },
                            }
                            offset = 0
                            for index in indices:
                                self.assertNotIn(index, restored, "sample was packed more than once")
                                restored[index] = {
                                    key: value[:, offset : offset + lengths[index]].clone()
                                    for key, value in fields.items()
                                }
                                offset += lengths[index]
                            self.assertEqual(seq.num_padding, worker.config.pack_max_length - offset)
                            for value in fields.values():
                                self.assertEqual(value.shape[1], worker.config.pack_max_length)
                self.assertEqual(sorted(restored), list(range(len(states))))
                for index, original in enumerate(originals):
                    for key, expected in original.items():
                        with self.subTest(sample=index, field=key):
                            torch.testing.assert_close(restored[index][key], expected, rtol=0, atol=0)
                            # Packing must also leave its source fields untouched.
                            source = (
                                getattr(samples[index].seq_ctx, key)
                                if key in ("input_ids", "position_ids")
                                else _sample_fields(samples[index])[key]
                            )
                            torch.testing.assert_close(source, expected, rtol=0, atol=0)

    def test_single_item_without_padding_preserves_fields(self):
        # 单样本满包仍经过 cat，保留字段和值并统一 packed 表示。
        worker = _make_worker()
        state = _make_state(
            uid=1,
            prompt_ids=[10, 11, 12, 10, 11, 12, 10, 11, 12, 10, 11, 12, 10, 11, 12, 10],
            response_ids=[20],
        )
        samples = _convert(worker, [state], [1.0])
        self.assertEqual(samples[0].seq_ctx.input_ids.shape[1], worker.config.pack_max_length)

        step_batches = _pack(worker, samples, [[[0]]])

        seq_ctx, _ = step_batches[0][0]
        torch.testing.assert_close(seq_ctx.input_ids, samples[0].seq_ctx.input_ids)
        self.assertEqual(seq_ctx.num_padding, 0)

    def test_empty_pack_materializes_all_padding(self):
        # 调度占位空 pack：整包都是 padding（全零 input_ids + labels=-100 + advantages=0）。
        worker = _make_worker()

        step_batches = _pack(worker, [], [[[]]])

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

        step_batches = _pack(worker, [], [[[]]], pack_loss_keys=["rollout_logprobs"])

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

        step_batches = _pack(worker, [], [[[]]])

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
        # 同一 pack 模板要求的 teacher loss 键必须在所有真实样本中存在，混出即 fail fast。
        worker = _make_worker()
        worker.config.distillation_config = _train_teacher_config()
        samples = [
            TrainSample(
                seq_ctx=SequenceContext.from_input_ids((torch.tensor([[1, 2]]),), device="cpu"),
                shifted_labels=torch.tensor([[-100, 2]]),
                rollout_logprobs=None,
                teacher_targets={"teacher_indices": torch.tensor([[0, 0]])},
                advantages=torch.zeros(1, 2),
            ),
            TrainSample(
                seq_ctx=SequenceContext.from_input_ids((torch.tensor([[3, 4]]),), device="cpu"),
                shifted_labels=torch.tensor([[3, 4]]),
                rollout_logprobs=None,
                teacher_targets={},
                advantages=torch.zeros(1, 2),
            ),
        ]

        with self.assertRaisesRegex(ValueError, "must be present on every sample"):
            _pack(worker, samples, [[[0, 1]]], pack_loss_keys=["teacher_indices"])

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
        samples = _convert(worker, [state], [1.0])

        with self.assertRaises(AssertionError):
            _pack(worker, samples, [[[0]]])

    def test_padding_seq_ctx_uses_3d_positions_when_configured(self):
        # use_3d_position_ids=True 时 padding 区 position_ids 为三轴 arange。
        worker = _make_worker()

        step_batches = _pack(worker, [], [[[]]], use_3d=True)

        position_ids = step_batches[0][0][0].position_ids
        self.assertEqual(tuple(position_ids.shape), (3, 1, worker.config.pack_max_length))
        torch.testing.assert_close(
            position_ids,
            torch.arange(worker.config.pack_max_length, dtype=torch.long).reshape(1, 1, -1).expand(3, -1, -1),
        )

    def test_padding_seq_ctx_gets_dummy_routed_experts(self):
        # has_routed_experts=True 时 padding 区补 dummy expert id，形状与真实 rollout_routed_experts 对齐。
        worker = _make_worker()

        step_batches = _pack(worker, [], [[[]]], has_routed_experts=True)

        routed_experts = step_batches[0][0][0].rollout_routed_experts[0]
        self.assertEqual(tuple(routed_experts.shape), (worker.config.pack_max_length, 1, 1))
        self.assertTrue(torch.all(routed_experts >= 0))
        self.assertTrue(torch.all(routed_experts < worker.config.model_cfg.n_routed_experts))

    def test_required_loss_field_missing_from_every_real_sample_raises(self):
        # 模板 teacher 键在全部真实样本上缺失同样 fail fast。
        worker = _make_worker()
        worker.config.distillation_config = _train_teacher_config()
        samples = _convert(worker, [_make_state()], [1.0])

        with self.assertRaisesRegex(ValueError, "Required pack field teacher_indices"):
            _pack(worker, samples, [[[0]]], pack_loss_keys=["teacher_indices"])

    def test_sampled_teacher_padding_preserves_real_targets_and_empty_dp(self):
        worker = _make_worker(
            _rollout_distillation_config(DistillationLossConfig(policy_loss_cfg={"loss_type": "vanilla"}))
        )
        state = _make_state(teacher_tokens=[20, 21, 22], teacher_logprobs=[-0.1, -0.2, -0.3])
        samples = _convert(worker, [state], [0.0])
        real_length = samples[0].seq_ctx.input_ids.shape[1]
        # Two DP ranks: one real sample and one scheduling-only empty pack.
        plans, _ = RLDataPacker(16, 2, 1, 1).pack([real_length])
        for dp_rank, plan in enumerate(plans):
            with self.subTest(dp_rank=dp_rank):
                seq, loss = _pack(worker, samples if dp_rank == 0 else [], plan)[0][0]
                kwargs = loss.loss_kwargs
                self.assertEqual(tuple(kwargs.teacher_logprobs.shape), (1, 16))
                self.assertEqual(kwargs.teacher_logprobs.device.type, "cpu")
                self.assertIsNone(kwargs.target_token_ids)
                self.assertEqual(seq.input_ids.shape[1], kwargs.shifted_labels.shape[1])
                start = real_length if dp_rank == 0 else 0
                if dp_rank == 0:
                    torch.testing.assert_close(
                        kwargs.teacher_logprobs[:, :real_length], samples[0].teacher_targets["teacher_logprobs"]
                    )
                self.assertTrue(torch.all(kwargs.teacher_logprobs[:, start:] == 0))
                self.assertTrue(torch.all(kwargs.shifted_labels[:, start:] == -100))
                self.assertTrue(torch.all(kwargs.advantages[:, start:] == 0))

    def test_short_pack_pads_topk_targets_and_rollout_logprobs(self):
        worker = _make_worker(
            _rollout_distillation_config(
                DistillationLossConfig(
                    policy_loss_cfg={"loss_type": "vanilla"},
                    loss_mode="forward_kl_topk",
                    use_policy_gradient=False,
                    top_k=2,
                )
            )
        )
        state = _make_state(
            logprobs=[-0.3, -0.4, -0.5],
            teacher_tokens=[[20, 30], [21, 31], [22, 32]],
            teacher_logprobs=[[-0.1, -0.2]] * 3,
        )
        samples = _convert(worker, [state], [0.0])
        real_length = samples[0].seq_ctx.input_ids.shape[1]
        seq, loss = _pack(worker, samples, [[[0]]], pack_loss_keys=["rollout_logprobs"])[0][0]
        originals = _sample_fields(samples[0])
        for key in ("shifted_labels", "advantages", "rollout_logprobs", "teacher_logprobs", "target_token_ids"):
            with self.subTest(key=key):
                packed = getattr(loss.loss_kwargs, key)
                original = originals[key]
                self.assertEqual(packed.shape[1], seq.input_ids.shape[1])
                self.assertEqual(packed.device.type, "cpu")
                torch.testing.assert_close(packed[:, :real_length], original)
                pad_value = -100 if key == "shifted_labels" else 0
                self.assertTrue(torch.all(packed[:, real_length:] == pad_value))

    def test_short_pack_pads_training_teacher_indices(self):
        worker = _make_worker()
        loss_config = DistillationLossConfig(policy_loss_cfg={"loss_type": "vanilla"})
        worker.config.loss_cfg = loss_config
        worker.config.distillation_config = _train_teacher_config()
        sample = TrainSample(
            seq_ctx=SequenceContext.from_input_ids((torch.tensor([[1, 2, 3]]),), device="cpu"),
            shifted_labels=torch.tensor([[-100, 2, 3]]),
            rollout_logprobs=None,
            teacher_targets={"teacher_indices": torch.tensor([[2, 2, 2]])},
            advantages=torch.tensor([[0.0, 1.0, 1.0]]),
        )
        _, packed = _pack(worker, [sample], [[[0]]], pack_loss_keys=["teacher_indices"])[0][0]
        torch.testing.assert_close(packed.loss_kwargs.teacher_indices[:, :3], torch.tensor([[2, 2, 2]]))
        self.assertTrue(torch.all(packed.loss_kwargs.teacher_indices[:, 3:] == -1))
        self.assertEqual(tuple(packed.loss_kwargs.teacher_indices.shape), (1, 16))

    def test_single_full_pack_materializes_single_ref_and_segment_refs(self):
        # resolve_routed_experts 把 ref / ref-list 物化为 pack 级张量；segmented 保留 [1:] 语义。
        for segmented in (False, True):
            with self.subTest(segmented=segmented):
                worker = _make_worker(
                    model_cfg=SimpleNamespace(n_routed_experts=8, num_hidden_layers=1, num_experts_per_tok=1)
                )
                ref = ray.ObjectRef(b"a" * 28)
                payload = [ref] if segmented else ref
                state = _make_state(prompt_ids=list(range(16)), response_ids=[20], routed_experts=payload)
                samples = _convert(worker, [state], [1.0])
                seq, _ = _pack(worker, samples, [[[0]]], has_routed_experts=True)[0][0]
                self.assertEqual(seq.rollout_routed_experts, [payload])
                values = (np.arange(17 if segmented else 16) % 8).astype(np.int32).reshape(-1, 1)
                with patch.object(ray, "get", return_value=values):
                    resolve_routed_experts(seq, model_cfg=worker.config.model_cfg, pack_max_length=16)
                expected = values[1:] if segmented else values
                np.testing.assert_array_equal(seq.rollout_routed_experts.numpy().reshape(-1, 1), expected)

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
        self.assertEqual(topk_worker.actor._configured_teacher_loss_keys(), ["teacher_logprobs", "target_token_ids"])

        sampled_worker = _make_worker(
            _rollout_distillation_config(
                DistillationLossConfig(
                    policy_loss_cfg={"loss_type": "vanilla"},
                    loss_mode="k1",
                    use_policy_gradient=True,
                )
            )
        )
        self.assertEqual(sampled_worker.actor._configured_teacher_loss_keys(), ["teacher_logprobs"])

        train_teacher_worker = _make_worker()
        train_teacher_worker.config.distillation_config = _train_teacher_config()
        self.assertEqual(train_teacher_worker.actor._configured_teacher_loss_keys(), ["teacher_indices"])

        self.assertEqual(_make_worker().actor._configured_teacher_loss_keys(), [])


if __name__ == "__main__":
    unittest.main()
