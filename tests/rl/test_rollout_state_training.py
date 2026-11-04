"""RolloutState owns its training-field contract (S1).

覆盖的行为:
- TestSetTrainingFields: 首次发布、非法输入不留半更新、prompt 长度推导
- TestApplyResponseMask: 重复 mask、语义洞保留、无显式 prompt 时长度稳定
- TestResetResponse: 训练字段与摘要一起失效、prompt 上下文保留、可重新生成
- TestSerialization: model_dump / model_copy / replay debug snapshot 保留摘要
"""

import unittest

from xtuner.v1.data_proto.rl_data import RolloutState, Status, TrainMeta


def make_state(**kwargs) -> RolloutState:
    state = RolloutState(
        rollout_id=1,
        group_id=1,
        message=[{"role": "user", "content": "prompt"}],
        prompt_ids=[10, 11],
        response_ids=[20, 21, 22],
        status=Status.COMPLETED,
        **kwargs,
    )
    state.response_model_steps = [1, 1, 1]
    return state


class TestSetTrainingFields(unittest.TestCase):
    def test_publishes_fields_length_and_meta(self):
        state = make_state()
        state.set_training_fields(
            input_ids=[10, 11, 20, 21, 22],
            labels=[-100, -100, 20, 21, 22],
            logprobs=[0.0, 0.0, -0.1, -0.2, -0.3],
        )

        self.assertEqual(state.num_tokens, 4)
        self.assertEqual(
            state.train_meta,
            TrainMeta(
                prompt_length=2,
                response_length=3,
                supervised_tokens=3,
                position_layout="1d",
                has_rollout_logprobs=True,
                has_routed_experts=False,
                segment_index=None,
            ),
        )
        state.validate_training()

    def test_invalid_input_leaves_no_partial_update(self):
        state = make_state()
        with self.assertRaises(ValueError):
            state.set_training_fields(input_ids=[10, 11], labels=[-100], logprobs=None)

        self.assertIsNone(state.input_ids)
        self.assertIsNone(state.labels)
        self.assertIsNone(state.num_tokens)
        self.assertIsNone(state.train_meta)

    def test_prompt_length_falls_back_to_supervision_count(self):
        # 无显式 prompt_ids 时，prompt 长度由 shift 后的非监督位推导。
        state = RolloutState(
            rollout_id=1,
            group_id=1,
            message=[],
            response_ids=[20, 21],
            status=Status.COMPLETED,
        )
        state.response_model_steps = [1, 1]
        state.set_training_fields(input_ids=[10, 20, 21], labels=[-100, 20, 21], logprobs=None)

        self.assertEqual(state.train_meta.prompt_length, 0)
        self.assertEqual(state.train_meta.supervised_tokens, 2)

    def test_validate_detects_stale_meta(self):
        state = make_state()
        state.set_training_fields(input_ids=[10, 11, 20, 21, 22], labels=[-100, -100, 20, 21, 22], logprobs=None)
        state.num_tokens = 99

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            state.validate_training()

    def test_validate_detects_version_length_mismatch(self):
        state = make_state()
        state.set_training_fields(input_ids=[10, 11, 20, 21, 22], labels=[-100, -100, 20, 21, 22], logprobs=None)
        state.response_model_steps = [1, 1]

        with self.assertRaisesRegex(ValueError, "response_model_steps length"):
            state.validate_training()


class TestApplyResponseMask(unittest.TestCase):
    def _published(self, *, prompt_ids=None):
        state = RolloutState(
            rollout_id=1,
            group_id=1,
            message=[],
            prompt_ids=prompt_ids,
            response_ids=[20, 21, 22],
            status=Status.COMPLETED,
        )
        state.response_model_steps = [1, 1, 1]
        state.set_training_fields(
            input_ids=[10, 11, 20, 21, 22],
            labels=[-100, -100, 20, 21, 22],
            logprobs=None,
        )
        return state

    def test_masks_only_unkept_response_positions(self):
        state = self._published()
        state.apply_response_mask([True, False, True])

        self.assertEqual(state.labels, [-100, -100, 20, -100, 22])
        self.assertEqual(state.train_meta.supervised_tokens, 2)
        self.assertEqual(state.num_tokens, 4)
        state.validate_training()

    def test_repeated_mask_is_idempotent(self):
        state = self._published()
        state.apply_response_mask([True, False, True])
        labels, meta = list(state.labels or []), state.train_meta
        state.apply_response_mask([True, False, True])

        self.assertEqual(state.labels, labels)
        self.assertEqual(state.train_meta, meta)

    def test_mask_does_not_restore_semantic_holes(self):
        # 语义洞（非 staleness）在 mask 为 True 时也保持 -100。
        state = self._published()
        state.labels = [-100, -100, 20, -100, 22]
        state.train_meta = state._build_train_meta(input_ids=[10, 11, 20, 21, 22], labels=state.labels, logprobs=None)
        state.apply_response_mask([True, True, True])

        self.assertEqual(state.labels, [-100, -100, 20, -100, 22])
        self.assertEqual(state.train_meta.supervised_tokens, 2)

    def test_prompt_length_is_stable_without_explicit_prompt(self):
        state = self._published(prompt_ids=None)
        prompt_length = state.train_meta.prompt_length
        state.apply_response_mask([False, False, False])

        self.assertEqual(state.train_meta.prompt_length, prompt_length)
        self.assertEqual(state.num_tokens, 4)
        state.validate_training()

    def test_wrong_mask_length_raises(self):
        state = self._published()
        with self.assertRaisesRegex(ValueError, "align with response_ids"):
            state.apply_response_mask([True])


class TestResetResponse(unittest.TestCase):
    def test_clears_training_fields_and_keeps_prompt_context(self):
        state = make_state(mm_info={"image_grid_thw": None})
        state.set_training_fields(input_ids=[10, 11, 20, 21, 22], labels=[-100, -100, 20, 21, 22], logprobs=None)
        state.reward = {"score": 1.0}

        state.reset_response()

        self.assertIsNone(state.input_ids)
        self.assertIsNone(state.labels)
        self.assertIsNone(state.num_tokens)
        self.assertIsNone(state.train_meta)
        self.assertEqual(state.response_ids, [])
        self.assertEqual(state.response_model_steps, [])
        self.assertEqual(state.tokens, [10, 11])
        self.assertIsNotNone(state.mm_info)

    def test_regenerated_fields_can_be_published_again(self):
        state = make_state()
        state.set_training_fields(input_ids=[10, 11, 20, 21, 22], labels=[-100, -100, 20, 21, 22], logprobs=None)
        state.reset_response()
        state.response_ids = [30, 31]
        state.response_model_steps = [2, 2]
        state.set_training_fields(input_ids=[10, 11, 30, 31], labels=[-100, -100, 30, 31], logprobs=None)

        self.assertEqual(state.train_meta.response_length, 2)
        state.validate_training()


class TestSerialization(unittest.TestCase):
    def test_model_dump_and_rebuild_keep_meta(self):
        state = make_state()
        state.set_training_fields(input_ids=[10, 11, 20, 21, 22], labels=[-100, -100, 20, 21, 22], logprobs=None)

        dumped = state.model_dump()
        rebuilt = RolloutState.model_validate(dumped)

        self.assertEqual(rebuilt.train_meta, state.train_meta)
        rebuilt.validate_training()

    def test_replay_debug_snapshot_keeps_meta(self):
        from xtuner.v1.rl.replay_buffer import _restore_nested_objectrefs, _snapshot_nested_objectrefs

        state = make_state()
        state.set_training_fields(input_ids=[10, 11, 20, 21, 22], labels=[-100, -100, 20, 21, 22], logprobs=None)

        restored = _restore_nested_objectrefs(_snapshot_nested_objectrefs(state))

        self.assertEqual(restored.train_meta, state.train_meta)
        restored.validate_training()


class TestRayRoundTrip(unittest.TestCase):
    def test_ray_round_trip_preserves_train_meta_and_methods(self):
        import ray

        started = not ray.is_initialized()
        if started:
            ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)
        try:
            state = make_state()
            state.set_training_fields(
                input_ids=[10, 11, 20, 21, 22],
                labels=[-100, -100, 20, 21, 22],
                logprobs=None,
            )

            restored = ray.get(ray.put(state))

            self.assertIsInstance(restored, RolloutState)
            self.assertEqual(restored.train_meta, state.train_meta)
            restored.validate_training()
            restored.apply_response_mask([True, False, True])
            self.assertEqual(restored.labels, [-100, -100, 20, -100, 22])
        finally:
            if started:
                ray.shutdown()
