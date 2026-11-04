"""TrainBatch materialization contract tests.

Covered behavior:
- shift layout and per-sample advantage alignment (regression: a one-token
  advantage mismatch used to shift the whole pack).
- mixed-batch 3D MRoPE axes; VLM multimodal fields and 3D positions preserved.
- top-k / sampled teacher targets aligned to the shifted layout.
- pack concatenation along the sequence dimension and the padding value table
  (labels=-100, advantages/rollout_logprobs=0, teacher_indices=-1, top-k
  teacher tensors zero-padded, 3D positions arange, routed experts dummy ids).
- template keys materialize on empty packs; mixed presence fails fast.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from xtuner.v1.data_proto.rl_data import RolloutState, Status, TeacherTargets
from xtuner.v1.rl.distillation import DistillationConfig, DistillationTrainerAdapter, RolloutTeacherConfig
from xtuner.v1.rl.loss import DistillationLossConfig, GRPOLossConfig
from xtuner.v1.rl.trainer.actor_worker import ActorWorker
from xtuner.v1.rl.trainer.pack import RankPlan, RLDataPacker, TokenBatchProto
from xtuner.v1.rl.trainer.train_batch import TrainBatch, get_train_seq_ctx


def _rollout_distillation_config(loss_config: DistillationLossConfig) -> DistillationConfig:
    return DistillationConfig(
        loss_config=loss_config,
        teachers=[RolloutTeacherConfig(name="teacher", endpoints=["http://teacher"])],
        data_source_teacher_map={"agent_math": "teacher"},
    )


def _worker_config(distillation_config=None, *, pack_max_length: int = 16):
    loss_cfg = (
        distillation_config.loss_config
        if distillation_config is not None
        else GRPOLossConfig(policy_loss_cfg={"loss_type": "vanilla"})
    )
    return SimpleNamespace(
        pack_max_length=pack_max_length,
        loss_cfg=loss_cfg,
        distillation_config=distillation_config,
        model_cfg=SimpleNamespace(n_routed_experts=8, num_hidden_layers=1, num_experts_per_tok=1),
    )


def _batch_attr(use_3d: bool = False, pack_loss_keys: list[str] | None = None, has_routed_experts: bool = False):
    return {
        "rollout_idx": 0,
        "use_3d_position_ids": use_3d,
        "pack_loss_keys": pack_loss_keys or [],
        "has_routed_experts": has_routed_experts,
    }


def _teacher_loss_keys(config: DistillationConfig | None) -> list[str]:
    if config is None:
        return []
    keys: list[str] = []
    if config.rollout_teachers:
        keys.append("teacher_logprobs")
        if not config.loss_config.uses_sampled_token_targets:
            keys.append("target_token_ids")
    if config.train_teachers:
        keys.append("teacher_indices")
    return keys


def _make_batch(
    states: list[RolloutState],
    advantages: list[float] | None = None,
    *,
    distillation_config: DistillationConfig | None = None,
    pack_max_length: int | None = None,
    use_3d: bool = False,
    pack_loss_keys: list[str] | None = None,
    has_routed_experts: bool = False,
) -> TrainBatch:
    if pack_max_length is None:
        pack_max_length = sum(len(state.labels or []) - 1 for state in states) or 16
    config = _worker_config(distillation_config, pack_max_length=pack_max_length)
    batch = TrainBatch(
        states,
        RankPlan(global_indices=tuple(range(len(states))), steps=((tuple(range(len(states))),),)),
        _batch_attr(use_3d, pack_loss_keys, has_routed_experts),  # type: ignore[arg-type]
        DistillationTrainerAdapter(distillation_config),
        pack_max_length=pack_max_length,
        model_cfg=config.model_cfg,  # type: ignore[arg-type]
        teacher_loss_keys=_teacher_loss_keys(distillation_config),
        distillation_config=distillation_config,
    )
    if advantages is not None:
        batch.install_advantages(advantages)
        batch.require_targets("actor")
    return batch


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
    return state


class TestTrainBatchMaterialize(unittest.TestCase):
    def test_text_path_shifts_and_aligns_loss_inputs(self):
        state = _make_state(
            response_ids=[20, 21, 22],
            logprobs=[0.1, 0.2, 0.3],
            supervised_mask=[1, 0, 1],
            routed_experts=np.array([[1, 2], [3, 4]]),
        )
        batch = _make_batch([state], [1.5], pack_loss_keys=["rollout_logprobs"])

        pack = batch.materialize((0,))

        seq_ctx = pack.full_data["seq_ctx"]
        self.assertEqual(seq_ctx.input_ids.tolist(), [[10, 11, 12, 20, 21]])
        self.assertEqual(pack.full_data["shifted_labels"].tolist(), [[-100, -100, 20, -100, 22]])
        torch.testing.assert_close(
            pack.full_data["advantages"], torch.tensor([[0.0, 0.0, 1.5, 0.0, 1.5]], dtype=torch.float32)
        )
        torch.testing.assert_close(
            pack.full_data["rollout_logprobs"], torch.tensor([[0.0, 0.0, 0.1, 0.2, 0.3]], dtype=torch.float32)
        )
        self.assertEqual(pack.full_data["shifted_labels"].device.type, "cpu")
        np.testing.assert_array_equal(seq_ctx.rollout_routed_experts[0], state.routed_experts)

    def test_mixed_batch_extends_text_positions_to_3d(self):
        state = _make_state(response_ids=[20, 21])
        batch = _make_batch([state], [0.0], use_3d=True)

        position_ids = batch.materialize((0,)).full_data["seq_ctx"].position_ids

        self.assertEqual(tuple(position_ids.shape), (3, 1, 4))
        torch.testing.assert_close(position_ids, torch.arange(4, dtype=torch.long).reshape(1, 1, -1).expand(3, -1, -1))

    def test_vlm_keeps_mm_fields_and_3d_positions(self):
        pixel_values = np.ones((1, 2, 3), dtype=np.float32)
        image_grid_thw = np.array([[1, 2, 3]], dtype=np.int32)
        routed_experts = np.array([[1, 2], [3, 4]])
        state = _make_state(
            prompt_ids=[100, 101],
            response_ids=[102, 103],
            position_ids=np.array([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=np.int64),
            mm_info={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            routed_experts=routed_experts,
        )
        batch = _make_batch([state], [0.25], use_3d=True)

        seq_ctx = batch.materialize((0,)).full_data["seq_ctx"]
        self.assertEqual(seq_ctx.input_ids.tolist(), [[100, 101, 102]])
        self.assertEqual(tuple(seq_ctx.position_ids.shape), (3, 1, 3))
        np.testing.assert_array_equal(seq_ctx.pixel_values[0], pixel_values)
        self.assertEqual(seq_ctx.image_grid_thw.tolist(), [[1, 2, 3]])
        np.testing.assert_array_equal(seq_ctx.rollout_routed_experts[0], routed_experts)

    def test_get_train_seq_ctx_mrope_continues_from_global_amax(self):
        input_ids = torch.tensor([[10, 11, 12, 20, 21, 22]])
        position_ids = np.array([[[0, 1]], [[0, 60]], [[0, 5]]], dtype=np.int64)

        seq_ctx = get_train_seq_ctx(input_ids, position_ids)

        self.assertEqual(tuple(seq_ctx.position_ids.shape), (3, 1, 6))
        torch.testing.assert_close(seq_ctx.position_ids[:, 0, 2:], torch.tensor([[61, 62, 63, 64]]).expand(3, -1))

    def test_topk_teacher_targets_align_shifted_layout(self):
        config = _rollout_distillation_config(
            DistillationLossConfig(
                policy_loss_cfg={"loss_type": "vanilla"},
                loss_mode="forward_kl_topk",
                use_policy_gradient=False,
                top_k=2,
            )
        )
        state = _make_state(
            input_ids=[10, 11, 20, 21, 22],
            labels=[-100, -100, 20, -100, 22],
            logprobs=[0.0, -0.1, -0.2, -0.3, -0.4],
            teacher_tokens=[[100, 101], [102, 103], [104, 105]],
            teacher_logprobs=[[-0.5, -0.6], [-0.7, -0.8], [-0.9, -1.0]],
        )
        batch = _make_batch([state], [0.0], distillation_config=config)

        pack = batch.materialize((0,))

        self.assertEqual(pack.full_data["shifted_labels"].tolist(), [[-100, 20, -100, 22]])
        self.assertEqual(pack.full_data["target_token_ids"].tolist(), [[[0, 0], [100, 101], [102, 103], [104, 105]]])
        torch.testing.assert_close(
            pack.full_data["teacher_logprobs"],
            torch.tensor([[[0.0, 0.0], [-0.5, -0.6], [-0.7, -0.8], [-0.9, -1.0]]], dtype=torch.float32),
        )

    def test_topk_targets_include_masked_response_rows(self):
        # labels=-100 的 response 行也保留 teacher 行（loss 侧自行忽略），只在前面补零行。
        config = _rollout_distillation_config(
            DistillationLossConfig(
                policy_loss_cfg={"loss_type": "vanilla"},
                loss_mode="forward_kl_topk",
                use_policy_gradient=False,
                top_k=2,
            )
        )
        state = _make_state(
            response_ids=[20, 21, 22],
            supervised_mask=[0, 1, 1],
            teacher_tokens=[[102, 103], [104, 105]],
            teacher_logprobs=[[-0.7, -0.8], [-0.9, -1.0]],
        )
        batch = _make_batch([state], [0.0], distillation_config=config)

        pack = batch.materialize((0,))
        self.assertEqual(pack.full_data["shifted_labels"].tolist(), [[-100, -100, -100, 21, 22]])
        self.assertEqual(
            pack.full_data["target_token_ids"].tolist(),
            [[[0, 0], [0, 0], [0, 0], [102, 103], [104, 105]]],
        )
        torch.testing.assert_close(
            pack.full_data["teacher_logprobs"],
            torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [-0.7, -0.8], [-0.9, -1.0]]], dtype=torch.float32),
        )

    def test_sampled_targets_have_no_token_ids(self):
        config = _rollout_distillation_config(
            DistillationLossConfig(
                policy_loss_cfg={"loss_type": "vanilla"},
                loss_mode="k1",
                use_policy_gradient=True,
            )
        )
        state = _make_state(
            response_ids=[20, 21], logprobs=[0.1, 0.2], teacher_tokens=[20, 21], teacher_logprobs=[-0.7, -0.9]
        )
        batch = _make_batch([state], [0.0], distillation_config=config)

        pack = batch.materialize((0,))
        self.assertNotIn("target_token_ids", pack.full_data)
        torch.testing.assert_close(
            pack.full_data["teacher_logprobs"], torch.tensor([[0.0, 0.0, -0.7, -0.9]], dtype=torch.float32)
        )


class TestTrainBatchPack(unittest.TestCase):
    def test_packs_concatenate_along_sequence_and_pad(self):
        first = _make_state(uid=1, response_ids=[20, 21], logprobs=[0.1, 0.2])
        second = _make_state(uid=2, prompt_ids=[30, 31], response_ids=[40], logprobs=[-0.3])
        batch = _make_batch([first, second], [1.0, 2.0], pack_max_length=16)

        pack = batch.materialize((0, 1))

        seq_ctx = pack.full_data["seq_ctx"]
        self.assertEqual(seq_ctx.input_ids.shape[1], 16)
        self.assertEqual(seq_ctx.num_padding, 16 - 6)
        self.assertEqual(pack.full_data["shifted_labels"][0, :4].tolist(), [-100, -100, 20, 21])
        self.assertEqual(pack.full_data["shifted_labels"][0, 4:6].tolist(), [-100, 40])
        torch.testing.assert_close(pack.full_data["advantages"][0, :6], torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 2.0]))
        self.assertTrue(torch.all(pack.full_data["shifted_labels"][0, 6:] == -100))
        self.assertTrue(torch.all(pack.full_data["advantages"][0, 6:] == 0))

    def test_pack_fields_round_trip_to_original_samples(self):
        config = _rollout_distillation_config(
            DistillationLossConfig(
                policy_loss_cfg={"loss_type": "vanilla"},
                loss_mode="forward_kl_topk",
                use_policy_gradient=False,
                top_k=2,
            )
        )
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
        originals = {}
        for index, state in enumerate(states):
            single = _make_batch(
                [state], [float(index + 1)], distillation_config=config, pack_max_length=lengths[index]
            )
            pack = single.materialize((0,))
            originals[index] = {
                "shifted_labels": pack.full_data["shifted_labels"].clone(),
                "advantages": pack.full_data["advantages"].clone(),
                "teacher_logprobs": pack.full_data["teacher_logprobs"].clone(),
            }
        for strategy in ("legacy", "greedy", "balance", "native"):
            with self.subTest(strategy=strategy):
                with patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", True):
                    packed_plan, _ = RLDataPacker(10, 3, 1, 2, pack_strategy=strategy, pack_seed=7).pack(lengths)
                restored = {}
                for dp_plan in packed_plan:
                    for step in dp_plan:
                        for pack_indices in step:
                            if not pack_indices:
                                continue
                            local_states = [states[index] for index in pack_indices]
                            batch = _make_batch(
                                local_states,
                                [states[index].reward["score"] for index in pack_indices],
                                distillation_config=config,
                                pack_max_length=10,
                            )
                            pack = batch.materialize(tuple(range(len(local_states))))
                            for offset, (index, row) in enumerate(
                                batch.unpack_rows(pack, pack.full_data["shifted_labels"])
                            ):
                                restored[pack_indices[offset]] = row.clone()
                self.assertEqual(sorted(restored), list(range(len(states))))
                for index, restored_labels in restored.items():
                    torch.testing.assert_close(restored_labels, originals[index]["shifted_labels"])

    def test_single_item_without_padding_preserves_fields(self):
        state = _make_state(
            prompt_ids=[10, 11, 12] * 5 + [10],
            response_ids=[20],
        )
        batch = _make_batch([state], [1.0], pack_max_length=16)
        pack = batch.materialize((0,))

        self.assertEqual(pack.full_data["seq_ctx"].num_padding, 0)
        self.assertEqual(pack.full_data["shifted_labels"].shape[1], 16)

    def test_empty_pack_materializes_all_padding(self):
        batch = _make_batch([], pack_max_length=16)
        pack = batch.materialize(())

        seq_ctx = pack.full_data["seq_ctx"]
        self.assertEqual(seq_ctx.input_ids.shape[1], 16)
        self.assertEqual(seq_ctx.num_padding, 16)
        self.assertTrue(torch.all(seq_ctx.input_ids == 0))
        self.assertTrue(torch.all(pack.full_data["shifted_labels"] == -100))
        self.assertTrue(torch.all(pack.full_data["advantages"] == 0))

    def test_rollout_logprobs_template_key_on_empty_pack(self):
        batch = _make_batch([], pack_max_length=16, pack_loss_keys=["rollout_logprobs"])
        pack = batch.materialize(())

        torch.testing.assert_close(pack.full_data["rollout_logprobs"], torch.zeros(1, 16, dtype=torch.float32))

    def test_teacher_template_keys_padded_on_empty_pack(self):
        config = _rollout_distillation_config(
            DistillationLossConfig(
                policy_loss_cfg={"loss_type": "vanilla"},
                loss_mode="forward_kl_topk",
                use_policy_gradient=False,
                top_k=2,
            )
        )
        batch = _make_batch([], pack_max_length=16, distillation_config=config)
        pack = batch.materialize(())

        torch.testing.assert_close(pack.full_data["teacher_logprobs"], torch.zeros(1, 16, 2, dtype=torch.float32))
        torch.testing.assert_close(pack.full_data["target_token_ids"], torch.zeros(1, 16, 2, dtype=torch.long))

    def test_mixed_presence_within_pack_raises(self):
        first = _make_state(uid=1, logprobs=[0.1, 0.2, 0.3])
        second = _make_state(uid=2)
        second.logprobs = None
        batch = _make_batch([first, second], [1.0, 2.0], pack_loss_keys=["rollout_logprobs"])

        with self.assertRaisesRegex(ValueError, "rollout_logprobs"):
            batch.materialize((0, 1))

    def test_oversize_pack_raises(self):
        long_ids = list(range(100, 120))
        state = _make_state(input_ids=long_ids, labels=[-100] * 19 + [42], logprobs=[0.0] * 20)
        batch = _make_batch([state], [1.0], pack_max_length=16)

        with self.assertRaisesRegex(ValueError, "exceed pack_max_length"):
            batch.materialize((0,))

    def test_padding_seq_ctx_uses_3d_positions_when_configured(self):
        batch = _make_batch([], pack_max_length=16, use_3d=True)
        position_ids = batch.materialize(()).full_data["seq_ctx"].position_ids

        self.assertEqual(tuple(position_ids.shape), (3, 1, 16))
        torch.testing.assert_close(position_ids[0, 0], torch.arange(16))

    def test_padding_seq_ctx_gets_dummy_routed_experts(self):
        batch = _make_batch([], pack_max_length=8, has_routed_experts=True)
        seq_ctx = batch.materialize(()).full_data["seq_ctx"]

        padding_experts = seq_ctx.rollout_routed_experts[0]
        self.assertEqual(tuple(padding_experts.shape), (8, 1, 1))
        self.assertTrue(torch.all(padding_experts < 8))

    def test_short_pack_pads_training_teacher_indices(self):
        batch = TrainBatch(
            [],
            RankPlan(global_indices=(), steps=(((),),)),
            _batch_attr(),  # type: ignore[arg-type]
            DistillationTrainerAdapter(None),
            pack_max_length=16,
            model_cfg=_worker_config().model_cfg,  # type: ignore[arg-type]
            teacher_loss_keys=["teacher_indices"],
            distillation_config=None,
        )
        teacher_indices = batch.materialize(()).full_data["teacher_indices"]

        self.assertEqual(tuple(teacher_indices.shape), (1, 16))
        self.assertTrue(torch.all(teacher_indices == -1))

    def test_unpack_rows_restores_sample_major_order(self):
        first = _make_state(uid=1, response_ids=[20, 21])
        second = _make_state(uid=2, response_ids=[30, 31, 32])
        batch = _make_batch([first, second], [1.0, 2.0], pack_max_length=9)
        pack = batch.materialize((0, 1))

        rows = batch.unpack_rows(pack, torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]))

        self.assertEqual([index for index, _ in rows], [0, 1])
        torch.testing.assert_close(rows[0][1], torch.tensor([[0.0, 1.0, 2.0, 3.0]]))
        torch.testing.assert_close(rows[1][1], torch.tensor([[4.0, 5.0, 6.0, 7.0, 8.0]]))


class TestConfiguredTeacherLossKeys(unittest.TestCase):
    def test_configured_teacher_loss_keys(self):
        topk = _rollout_distillation_config(
            DistillationLossConfig(
                policy_loss_cfg={"loss_type": "vanilla"},
                loss_mode="forward_kl_topk",
                use_policy_gradient=False,
                top_k=2,
            )
        )
        sampled = _rollout_distillation_config(
            DistillationLossConfig(
                policy_loss_cfg={"loss_type": "vanilla"},
                loss_mode="k1",
                use_policy_gradient=True,
            )
        )
        with_train_teacher = SimpleNamespace(
            rollout_teachers=[],
            train_teachers=[object()],
            loss_config=SimpleNamespace(uses_sampled_token_targets=True),
        )
        worker = ActorWorker.__new__(ActorWorker)
        for config, expected in (
            (None, []),
            (topk, ["teacher_logprobs", "target_token_ids"]),
            (sampled, ["teacher_logprobs"]),
            (with_train_teacher, ["teacher_indices"]),
        ):
            with self.subTest(config=config):
                worker.config = SimpleNamespace(distillation_config=config)
                self.assertEqual(worker.teacher_loss_keys(), expected)


if __name__ == "__main__":
    unittest.main()


class TestTrainBatchTargets(unittest.TestCase):
    """S4 target installation and trajectory export on the complete CPU batch."""

    def _batch(self) -> TrainBatch:
        state = _make_state(uid=1, response_ids=[20, 21], supervised_mask=[1, 0])
        return _make_batch([state], pack_max_length=4)

    def test_install_old_values_and_ppo_targets_round_trip(self):
        batch = self._batch()
        batch.install_old_values([torch.tensor([0.1, 0.2, 0.3, 0.4])])
        batch.install_ppo_targets(
            TokenBatchProto.concat([torch.tensor([1.0, 2.0, 3.0, 4.0])]),
            TokenBatchProto.concat([torch.tensor([5.0, 6.0, 7.0, 8.0])]),
        )
        batch.require_targets("critic")

        pack = batch.materialize((0,))
        torch.testing.assert_close(pack.full_data["old_values"], torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
        # advantages are zeroed at ignored positions; returns are installed verbatim.
        torch.testing.assert_close(pack.full_data["advantages"], torch.tensor([[0.0, 0.0, 3.0, 0.0]]))
        torch.testing.assert_close(pack.full_data["returns"], torch.tensor([[5.0, 6.0, 7.0, 8.0]]))

    def test_require_targets_rejects_incomplete_critic_targets(self):
        batch = self._batch()
        batch.install_old_values([torch.tensor([0.1, 0.2, 0.3, 0.4])])
        with self.assertRaisesRegex(ValueError, "returns"):
            batch.require_targets("critic")

    def test_install_ppo_targets_validates_lengths_before_writing(self):
        batch = self._batch()
        with self.assertRaisesRegex(ValueError, "returns row"):
            batch.install_ppo_targets(
                TokenBatchProto.concat([torch.tensor([1.0, 2.0, 3.0, 4.0])]),
                TokenBatchProto.concat([torch.tensor([1.0])]),
            )
        self.assertIsNone(batch._samples[0].advantages)
        self.assertIsNone(batch._samples[0].returns)

    def test_trajectory_data_exports_complete_rows_and_boundaries(self):
        first = _make_state(uid=1, response_ids=[20, 21], supervised_mask=[1, 0])
        second = _make_state(uid=2, response_ids=[30, 31, 32])
        second.reward = {"score": 2.0}
        batch = _make_batch([first, second], pack_max_length=9)
        batch.install_old_values([torch.tensor([0.1, 0.2, 0.3, 0.4]), torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])])

        trajectory = batch.trajectory_data()

        torch.testing.assert_close(trajectory["values"], torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        self.assertEqual(trajectory["scores"].tolist(), [1.0, 2.0])
        self.assertEqual(trajectory["mask"].tolist(), [False, False, True, False, False, False, True, True, True])
        self.assertEqual(trajectory["cu_seq_lens"].tolist(), [0, 4, 9])

    def test_trajectory_data_requires_reward_score(self):
        state = _make_state(uid=1, response_ids=[20, 21])
        state.reward = None
        batch = _make_batch([state], pack_max_length=4)
        batch.install_old_values([torch.tensor([0.1, 0.2, 0.3, 0.4])])

        with self.assertRaisesRegex(ValueError, "reward score"):
            batch.trajectory_data()
