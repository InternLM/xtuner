"""TrainBatch rollout-to-CPU-sample contract tests.

``TrainBatch`` is the only RolloutState -> CPU training data entry: it shifts
labels/logprobs once, broadcasts scalar advantages or installs token rows, unifies
the position layout, builds the seq_ctx and aligns teacher targets. Materialization
then concatenates and pads one pack. This file exercises that public API only; no
Ray, model or rollout backend is started.

Covered behavior:
- text samples: input_ids tail drop, shifted labels, shifted rollout logprobs.
- scalar advantage broadcast and token-row installation with masking.
- use_3d_position_ids broadcast and 1D default.
- VLM position extension, mm_info passthrough and routed_experts attachment.
- distillation teacher targets (sampled vs top-k).
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from xtuner.v1.data_proto.rl_data import RolloutState, Status, TeacherTargets
from xtuner.v1.rl.distillation import DistillationConfig, DistillationTrainerAdapter, RolloutTeacherConfig
from xtuner.v1.rl.loss import DistillationLossConfig
from xtuner.v1.rl.trainer.pack import RankPlan
from xtuner.v1.rl.trainer.train_batch import TrainBatch


def _batch_attr(use_3d: bool, pack_loss_keys: list[str] | None = None):
    return {
        "rollout_idx": 0,
        "use_3d_position_ids": use_3d,
        "pack_loss_keys": pack_loss_keys or [],
        "has_routed_experts": False,
    }


def _model_cfg():
    return SimpleNamespace(n_routed_experts=0, num_hidden_layers=1, num_experts_per_tok=1)


class TestTrainBatchContract(unittest.TestCase):
    def _make_batch(
        self,
        states: list[RolloutState],
        *,
        use_3d: bool = False,
        pack_max_length: int | None = None,
        pack_loss_keys: list[str] | None = None,
        distillation: DistillationTrainerAdapter | None = None,
        teacher_loss_keys: list[str] | None = None,
        distillation_config: DistillationConfig | None = None,
    ) -> TrainBatch:
        adapter = distillation if distillation is not None else DistillationTrainerAdapter(None)
        if pack_max_length is None:
            pack_max_length = sum(len(state.labels or []) - 1 for state in states)
        return TrainBatch(
            states,
            RankPlan(global_indices=tuple(range(len(states))), steps=((tuple(range(len(states))),),)),
            _batch_attr(use_3d, pack_loss_keys),  # type: ignore[arg-type]
            adapter,
            pack_max_length=pack_max_length,
            model_cfg=_model_cfg(),  # type: ignore[arg-type]
            teacher_loss_keys=teacher_loss_keys or [],
            distillation_config=distillation_config,
        )

    def _state(
        self,
        *,
        uid: int = 1,
        prompt_ids: list[int] | None = None,
        response_ids: list[int] | None = None,
        logprobs: list[float] | None = None,
        supervised_mask: list[int] | None = None,
        routed_experts=None,
        position_ids: np.ndarray | None = None,
        mm_info: dict | None = None,
        extra_fields: dict | None = None,
        input_ids: list[int] | None = None,
        labels: list[int] | None = None,
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
            status=Status.COMPLETED,
            finish_reason="stop",
            routed_experts=routed_experts,
            position_ids=position_ids,
            mm_info=mm_info,
            extra_fields=extra_fields or {},
            input_ids=input_ids,
            labels=labels,
            teacher_targets=(
                TeacherTargets(
                    kind="topk" if teacher_tokens and isinstance(teacher_tokens[0], list) else "sampled",
                    tokens=teacher_tokens,
                    logprobs=teacher_logprobs,
                )
                if teacher_tokens is not None and teacher_logprobs is not None
                else None
            ),
        )
        if input_ids is not None:
            return state
        state.input_ids = list(resolved_prompt_ids) + list(resolved_response_ids)
        if supervised_mask is None:
            state.labels = [-100] * len(resolved_prompt_ids) + list(resolved_response_ids)
        else:
            state.labels = [-100] * len(resolved_prompt_ids) + [
                resp_id if flag else -100 for resp_id, flag in zip(resolved_response_ids, supervised_mask)
            ]
        if state.logprobs is not None:
            state.logprobs = [0.0] * len(resolved_prompt_ids) + list(state.logprobs)
        return state

    def test_text_path_builds_shifted_train_sample(self):
        state = self._state(
            response_ids=[20, 21, 22],
            logprobs=[0.1, 0.2, 0.3],
            supervised_mask=[1, 0, 1],
        )
        batch = self._make_batch([state], pack_loss_keys=["rollout_logprobs"])
        batch.set_advantages([1.5])
        batch.require_targets("actor")

        pack = batch.materialize((0,))
        seq_ctx = pack.full_data["seq_ctx"]
        self.assertEqual(seq_ctx.input_ids.tolist(), [[10, 11, 12, 20, 21]])
        self.assertEqual(pack.full_data["shifted_labels"].tolist(), [[-100, -100, 20, -100, 22]])
        torch.testing.assert_close(
            pack.full_data["advantages"],
            torch.tensor([[0.0, 0.0, 1.5, 0.0, 1.5]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            pack.full_data["rollout_logprobs"],
            torch.tensor([[0.0, 0.0, 0.1, 0.2, 0.3]], dtype=torch.float32),
        )
        self.assertEqual(pack.lengths, (5,))
        for key in ("shifted_labels", "advantages", "rollout_logprobs"):
            self.assertEqual(pack.full_data[key].shape[1], seq_ctx.input_ids.shape[1])

    def test_advantages_follow_each_sample_scalar(self):
        first = self._state(uid=1, response_ids=[20, 21], supervised_mask=[1, 1])
        second = self._state(uid=2, response_ids=[30, 31], supervised_mask=[1, 0])
        batch = self._make_batch([first, second], pack_max_length=4)
        batch.set_advantages([1.5, -2.0])

        first_pack = batch.materialize((0,))
        second_pack = batch.materialize((1,))
        torch.testing.assert_close(
            first_pack.full_data["advantages"], torch.tensor([[0.0, 0.0, 1.5, 1.5]], dtype=torch.float32)
        )
        torch.testing.assert_close(
            second_pack.full_data["advantages"], torch.tensor([[0.0, 0.0, -2.0, 0.0]], dtype=torch.float32)
        )

    def test_text_positions_stay_1d_without_3d_flag(self):
        batch = self._make_batch([self._state(response_ids=[20, 21])])
        pack = batch.materialize((0,))

        torch.testing.assert_close(pack.full_data["seq_ctx"].position_ids, torch.arange(4).reshape(1, -1))

    def test_text_positions_broadcast_to_3d_in_mixed_batches(self):
        batch = self._make_batch([self._state(response_ids=[20, 21])], use_3d=True)
        pack = batch.materialize((0,))

        position_ids = pack.full_data["seq_ctx"].position_ids
        self.assertEqual(tuple(position_ids.shape), (3, 1, 4))
        torch.testing.assert_close(
            position_ids,
            torch.arange(4, dtype=torch.long).reshape(1, 1, -1).expand(3, -1, -1),
        )

    def test_vlm_position_extension_and_multimodal_passthrough(self):
        pixel_values = np.ones((1, 2, 3), dtype=np.float32)
        image_grid_thw = np.array([[1, 2, 3]], dtype=np.int32)
        routed_experts = np.array([[1, 2], [3, 4]])
        state = self._state(
            prompt_ids=[100, 101],
            response_ids=[102, 103],
            position_ids=np.array([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=np.int64),
            mm_info={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            routed_experts=routed_experts,
        )
        batch = self._make_batch([state], use_3d=True)
        pack = batch.materialize((0,))

        seq_ctx = pack.full_data["seq_ctx"]
        self.assertEqual(seq_ctx.input_ids.tolist(), [[100, 101, 102]])
        self.assertEqual(tuple(seq_ctx.position_ids.shape), (3, 1, 3))
        np.testing.assert_array_equal(seq_ctx.pixel_values[0], pixel_values)
        self.assertEqual(seq_ctx.image_grid_thw.tolist(), [[1, 2, 3]])
        np.testing.assert_array_equal(seq_ctx.rollout_routed_experts[0], routed_experts)

    def _distillation(
        self, loss_config: DistillationLossConfig
    ) -> tuple[DistillationTrainerAdapter, DistillationConfig]:
        config = DistillationConfig(
            loss_config=loss_config,
            teachers=[RolloutTeacherConfig(name="teacher", endpoints=["http://teacher"])],
            data_source_teacher_map={"agent_math": "teacher"},
        )
        return DistillationTrainerAdapter(config), config

    def test_topk_teacher_targets_land_in_pack_fields(self):
        adapter, config = self._distillation(
            DistillationLossConfig(
                policy_loss_cfg={"loss_type": "vanilla"},
                loss_mode="forward_kl_topk",
                use_policy_gradient=False,
                top_k=2,
            )
        )
        state = self._state(
            input_ids=[10, 11, 20, 21, 22],
            labels=[-100, -100, 20, -100, 22],
            logprobs=[0.0, -0.1, -0.2, -0.3, -0.4],
            teacher_tokens=[[100, 101], [102, 103], [104, 105]],
            teacher_logprobs=[[-0.5, -0.6], [-0.7, -0.8], [-0.9, -1.0]],
            extra_fields={"origin_data_source": "agent_math"},
        )
        batch = self._make_batch(
            [state],
            distillation=adapter,
            teacher_loss_keys=["teacher_logprobs", "target_token_ids"],
            distillation_config=config,
        )
        pack = batch.materialize((0,))

        self.assertEqual(pack.full_data["shifted_labels"].tolist(), [[-100, 20, -100, 22]])
        self.assertEqual(
            pack.full_data["target_token_ids"].tolist(),
            [[[0, 0], [100, 101], [102, 103], [104, 105]]],
        )
        torch.testing.assert_close(
            pack.full_data["teacher_logprobs"],
            torch.tensor([[[0.0, 0.0], [-0.5, -0.6], [-0.7, -0.8], [-0.9, -1.0]]], dtype=torch.float32),
        )

    def test_sampled_teacher_targets_omit_target_token_ids(self):
        adapter, config = self._distillation(
            DistillationLossConfig(
                policy_loss_cfg={"loss_type": "vanilla"},
                loss_mode="k1",
                use_policy_gradient=True,
            )
        )
        state = self._state(
            response_ids=[20, 21],
            logprobs=[0.1, 0.2],
            teacher_tokens=[20, 21],
            teacher_logprobs=[-0.7, -0.9],
            extra_fields={"origin_data_source": "agent_math"},
        )
        batch = self._make_batch(
            [state],
            distillation=adapter,
            teacher_loss_keys=["teacher_logprobs"],
            distillation_config=config,
        )
        pack = batch.materialize((0,))

        self.assertNotIn("target_token_ids", pack.full_data)
        torch.testing.assert_close(
            pack.full_data["teacher_logprobs"],
            torch.tensor([[0.0, 0.0, -0.7, -0.9]], dtype=torch.float32),
        )

    def test_token_advantages_mask_prompt_and_semantic_holes(self):
        from xtuner.v1.rl.trainer.pack import TokenBatchProto

        state = self._state(supervised_mask=[1, 0, 1])
        batch = self._make_batch([state])
        batch.set_advantages(TokenBatchProto.concat([torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])]))

        pack = batch.materialize((0,))
        torch.testing.assert_close(pack.full_data["advantages"], torch.tensor([[0.0, 0.0, 3.0, 0.0, 5.0]]))

    def test_token_advantages_require_shifted_length(self):
        from xtuner.v1.rl.trainer.pack import TokenBatchProto

        batch = self._make_batch([self._state()])
        with self.assertRaisesRegex(ValueError, "Token advantage length mismatch"):
            batch.set_advantages(TokenBatchProto.concat([torch.tensor([1.0])]))

    def test_empty_pack_materializes_as_padding(self):
        batch = self._make_batch([self._state()], pack_max_length=8)
        batch.set_advantages([1.0])
        pack = batch.materialize(())

        self.assertEqual(pack.indices, ())
        self.assertEqual(pack.lengths, ())
        self.assertEqual(pack.full_data["shifted_labels"].tolist(), [[-100] * 8])
        torch.testing.assert_close(pack.full_data["advantages"], torch.zeros(1, 8))

    def test_pack_pads_loss_fields_and_seq_ctx_together(self):
        batch = self._make_batch([self._state(response_ids=[20, 21])], pack_max_length=8)
        batch.set_advantages([1.0])
        pack = batch.materialize((0,))

        self.assertEqual(pack.full_data["seq_ctx"].input_ids.shape[1], 8)
        self.assertEqual(pack.full_data["shifted_labels"].shape[1], 8)
        self.assertEqual(pack.full_data["advantages"].shape[1], 8)

    def test_unpack_rows_restores_sample_rows(self):
        first = self._state(uid=1, response_ids=[20, 21])
        second = self._state(uid=2, response_ids=[30, 31, 32])
        batch = self._make_batch([first, second])
        pack = batch.materialize((0, 1))
        full_rows = torch.arange(5, dtype=torch.float32).reshape(1, -1)

        rows = batch.unpack_rows(pack, full_rows)

        self.assertEqual([index for index, _ in rows], [0, 1])
        torch.testing.assert_close(rows[0][1], torch.tensor([[0.0, 1.0, 2.0, 3.0]]))
        torch.testing.assert_close(rows[1][1], torch.tensor([[4.0]]))


if __name__ == "__main__":
    unittest.main()
