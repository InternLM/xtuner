import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import torch

from recipe.on_policy_distillation.build_teacher_server_commands import (
    build_teacher_launch_server_commands,
)
from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.rl.distillation import (
    DistillationConfig,
    DistillationTrainerAdapter,
    RolloutTeacherClient,
    RolloutTeacherConfig,
    TeacherTargetConfig,
    TrainTeacherManager,
    TrainTeacherOutputs,
    TrainTeacherTimings,
)
from xtuner.v1.rl.loss import DistillationLossConfig, DistillationLossKwargs
from xtuner.v1.rl.trainer.controller import TrainingController
from xtuner.v1.train.rl_trainer import BaseRLTrainer, BaseRLTrainerConfig


class TestDistillationRecipeConfig(unittest.TestCase):
    def test_teacher_launcher_reads_distillation_config(self) -> None:
        config_source = """
from xtuner.v1.rl.distillation import (
    DistillationConfig,
    RolloutTeacherConfig,
    RolloutTeacherLaunchConfig,
)
from xtuner.v1.rl.loss import DistillationLossConfig

loss_cfg = DistillationLossConfig(policy_loss_cfg={"loss_type": "vanilla"})
distillation_config = DistillationConfig(
    loss_config=loss_cfg,
    teachers=[
        RolloutTeacherConfig(
            name="teacher",
            launch_config=RolloutTeacherLaunchConfig(
                model_path="/models/teacher",
                num_workers=1,
                server_port=13141,
            ),
        )
    ],
    data_source_teacher_map={"math": "teacher"},
)
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "distillation_config.py"
            config_path.write_text(config_source)
            with patch.dict(
                "os.environ",
                {
                    "NODE_COUNT": "1",
                    "NODE_RANK": "0",
                    "PROC_PER_NODE": "2",
                    "WORKER_ALL_SOCKET_ADDRS": "127.0.0.1",
                },
            ):
                endpoint_map, student_num_workers, student_local_num_workers, records = (
                    build_teacher_launch_server_commands(str(config_path), "lmdeploy")
                )

        self.assertEqual(endpoint_map, {"teacher": ["http://127.0.0.1:13141"]})
        self.assertEqual(student_num_workers, 1)
        self.assertEqual(student_local_num_workers, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][:4], ["teacher[0]", "0", "1", "http://127.0.0.1:13141"])
        self.assertIn("lmdeploy", records[0][7:])

    def test_sampled_token_sampling_is_validated_by_trainer_config(self) -> None:
        loss_config = DistillationLossConfig(policy_loss_cfg={"loss_type": "vanilla"})
        distillation_config = DistillationConfig(
            loss_config=loss_config,
            teachers=[RolloutTeacherConfig(name="teacher", endpoints=["http://teacher"])],
            data_source_teacher_map={"math": "teacher"},
        )
        config = BaseRLTrainerConfig.model_construct(
            train_worker_cfg=SimpleNamespace(loss_cfg=loss_config, model_cfg=MagicMock()),
            agent_loop_manager_cfg=SimpleNamespace(
                tasks=SimpleNamespace(
                    task_name="math",
                    agent_loop_config=SimpleNamespace(sample_params=SampleParams(temperature=0.7)),
                )
            ),
            distillation_config=distillation_config,
            total_train_steps=1,
        )

        with self.assertRaisesRegex(ValueError, "Invalid sample_params for task 'math'.*temperature"):
            config._validate_sync_intervals()

    def test_distillation_trainer_validation_handles_multiple_tasks(self) -> None:
        loss_config = DistillationLossConfig(policy_loss_cfg={"loss_type": "vanilla"})
        distillation_config = DistillationConfig(
            loss_config=loss_config,
            teachers=[RolloutTeacherConfig(name="teacher", endpoints=["http://teacher"])],
            data_source_teacher_map={"math": "teacher", "code": "teacher"},
        )
        trainer_cfg = SimpleNamespace(
            train_worker_cfg=SimpleNamespace(loss_cfg=loss_config, model_cfg=MagicMock()),
            agent_loop_manager_cfg=SimpleNamespace(
                tasks=[
                    SimpleNamespace(
                        task_name="math",
                        agent_loop_config=SimpleNamespace(sample_params=SampleParams()),
                    ),
                    SimpleNamespace(
                        task_name="code",
                        agent_loop_config=SimpleNamespace(sample_params=SampleParams()),
                    ),
                ]
            ),
        )

        distillation_config.validate_trainer(trainer_cfg)

    def test_distillation_trainer_validation_rejects_missing_sample_params(self) -> None:
        loss_config = DistillationLossConfig(policy_loss_cfg={"loss_type": "vanilla"})
        distillation_config = DistillationConfig(
            loss_config=loss_config,
            teachers=[RolloutTeacherConfig(name="teacher", endpoints=["http://teacher"])],
            data_source_teacher_map={"math": "teacher"},
        )
        trainer_cfg = SimpleNamespace(
            train_worker_cfg=SimpleNamespace(loss_cfg=loss_config, model_cfg=MagicMock()),
            agent_loop_manager_cfg=SimpleNamespace(
                tasks=SimpleNamespace(
                    task_name="math",
                    agent_loop_config=SimpleNamespace(sample_params=None),
                )
            ),
        )

        with self.assertRaisesRegex(ValueError, "Task 'math' must configure sample_params"):
            distillation_config.validate_trainer(trainer_cfg)

    def test_distillation_trainer_validation_skips_sampling_constraints_for_topk(self) -> None:
        loss_config = DistillationLossConfig(
            policy_loss_cfg={"loss_type": "vanilla"},
            loss_mode="forward_kl_topk",
            use_policy_gradient=False,
            top_k=2,
        )
        distillation_config = DistillationConfig(
            loss_config=loss_config,
            teachers=[RolloutTeacherConfig(name="teacher", endpoints=["http://teacher"])],
            data_source_teacher_map={"math": "teacher"},
        )
        trainer_cfg = SimpleNamespace(
            train_worker_cfg=SimpleNamespace(loss_cfg=loss_config, model_cfg=MagicMock()),
            agent_loop_manager_cfg=SimpleNamespace(
                tasks=SimpleNamespace(
                    task_name="math",
                    agent_loop_config=SimpleNamespace(sample_params=SampleParams(temperature=0.7)),
                )
            ),
        )

        distillation_config.validate_trainer(trainer_cfg)

    def test_distillation_trainer_validation_rejects_mismatched_loss_config(self) -> None:
        distillation_loss_config = DistillationLossConfig(policy_loss_cfg={"loss_type": "vanilla"})
        worker_loss_config = DistillationLossConfig(
            policy_loss_cfg={"loss_type": "vanilla"},
            task_adv_weight=1.0,
        )
        distillation_config = DistillationConfig(
            loss_config=distillation_loss_config,
            teachers=[RolloutTeacherConfig(name="teacher", endpoints=["http://teacher"])],
            data_source_teacher_map={"math": "teacher"},
        )
        trainer_cfg = SimpleNamespace(
            train_worker_cfg=SimpleNamespace(loss_cfg=worker_loss_config, model_cfg=MagicMock()),
            agent_loop_manager_cfg=SimpleNamespace(tasks=SimpleNamespace()),
        )

        with self.assertRaisesRegex(ValueError, "train_worker_cfg.loss_cfg must be distillation_config.loss_config"):
            distillation_config.validate_trainer(trainer_cfg)

    def test_rollout_teacher_scorer_config_is_projected_from_distillation_config(self) -> None:
        loss_config = DistillationLossConfig(policy_loss_cfg={"loss_type": "vanilla"})
        distillation_config = DistillationConfig(
            loss_config=loss_config,
            teachers=[RolloutTeacherConfig(name="teacher", endpoints=["http://teacher"])],
            data_source_teacher_map={"math": "teacher"},
        )

        scorer_config = distillation_config.rollout_teacher_scorer_config

        self.assertIsNotNone(scorer_config)
        assert scorer_config is not None
        self.assertEqual(scorer_config.target_config, distillation_config.teacher_target_config)
        self.assertEqual([teacher.name for teacher in scorer_config.teachers], ["teacher"])
        self.assertEqual(scorer_config.data_source_teacher_map, {"math": "teacher"})
        self.assertTrue(scorer_config.target_config.uses_sampled_token_targets)
        self.assertIsNone(scorer_config.target_config.top_k)
        self.assertIsNone(distillation_config.train_teacher_manager_config)

    def test_distillation_trainer_adapter_uses_null_defaults_without_config(self) -> None:
        adapter = DistillationTrainerAdapter(None)

        self.assertIsNone(adapter.rollout_teacher_scorer_config)
        self.assertIsNone(adapter.train_teacher_manager_config)
        self.assertEqual(adapter.task_adv_weight, 1.0)
        self.assertIsNone(adapter.teacher_index_by_data_source)
        self.assertEqual(adapter.timing_scalars([]), {})

    def test_train_teacher_config_is_not_forwarded_to_agent_loop_manager(self) -> None:
        trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        trainer._distillation = DistillationTrainerAdapter(None)
        trainer._enable_evaluate = False
        trainer._resolve_total_train_steps = MagicMock()
        trainer.rollout_controller = MagicMock()
        trainer.logger = MagicMock()
        built_manager = MagicMock()
        manager_config = MagicMock()
        manager_config.build.return_value = built_manager
        config = SimpleNamespace(
            tokenizer_path="student",
            agent_loop_manager_cfg=manager_config,
            sync_weights_interval=1,
        )

        with patch("xtuner.v1.train.rl_trainer.AutoTokenizer.from_pretrained", return_value=MagicMock()):
            trainer._build_agent_loop_components(config, replay_buffer=MagicMock())

        self.assertIs(trainer.agent_loop_manager, built_manager)
        self.assertIsNone(manager_config.build.call_args.kwargs["rollout_teacher_scorer_config"])


class TestTrainTeacherTimings(unittest.TestCase):
    def test_lifecycle_timings_are_recorded_for_every_teacher(self) -> None:
        manager = TrainTeacherManager.__new__(TrainTeacherManager)
        first_teacher = MagicMock()
        second_teacher = MagicMock()
        timings = TrainTeacherTimings()

        with (
            patch(
                "xtuner.v1.rl.distillation.train_teacher_manager.time.perf_counter",
                side_effect=[1.0, 2.0, 3.0, 5.0, 6.0, 9.0, 10.0, 14.0, 15.0, 20.0, 21.0, 27.0],
            ),
            patch.object(manager, "_synchronize_device"),
            patch.object(manager, "_offload_to_cpu"),
        ):
            with manager._teacher_on_device(first_teacher, timings, "teacher_a"):
                pass
            with manager._teacher_on_device(second_teacher, timings, "teacher_b"):
                pass

        self.assertEqual(
            timings.to_dict(),
            {
                "teacher_a": {"compute": 2.0, "onload": 1.0, "offload": 3.0},
                "teacher_b": {"compute": 5.0, "onload": 4.0, "offload": 6.0},
            },
        )
        self.assertEqual((timings.compute, timings.onload, timings.offload), (7.0, 5.0, 9.0))

    def test_compute_teacher_outputs_writes_teacher_outputs_to_loss_contexts(self) -> None:
        manager = TrainTeacherManager.__new__(TrainTeacherManager)
        teacher_indices = torch.tensor([[0, 1]], dtype=torch.long)
        shifted_labels = torch.tensor([[-100, 10]], dtype=torch.long)
        teacher_logprobs = [torch.tensor([[-0.1, -0.2]])]
        target_token_ids = [torch.tensor([[[1, 2], [3, 4]]], dtype=torch.long)]
        timings = TrainTeacherTimings()
        seq_ctx = MagicMock()
        loss_ctx = SimpleNamespace(
            loss_kwargs=SimpleNamespace(
                teacher_indices=teacher_indices,
                shifted_labels=shifted_labels,
            )
        )

        with (
            patch.object(
                manager,
                "compute_logprobs",
                return_value=TrainTeacherOutputs(
                    teacher_logprobs=teacher_logprobs,
                    target_token_ids=target_token_ids,
                    timings=timings,
                ),
            ) as compute_logprobs,
            patch.object(manager, "offload_all_to_cpu") as offload_all_to_cpu,
        ):
            result = manager.compute_teacher_outputs([seq_ctx], [loss_ctx])

        self.assertIs(result, timings)
        compute_logprobs.assert_called_once_with(
            seq_ctx_list=[seq_ctx],
            shifted_labels_list=[shifted_labels],
            teacher_indices_list=[teacher_indices],
        )
        torch.testing.assert_close(loss_ctx.loss_kwargs.teacher_logprobs, teacher_logprobs[0])
        torch.testing.assert_close(loss_ctx.loss_kwargs.target_token_ids, target_token_ids[0])
        offload_all_to_cpu.assert_called_once_with()

    def test_distillation_loss_kwargs_carry_teacher_indices(self) -> None:
        loss_cfg = DistillationLossConfig(policy_loss_cfg={"loss_type": "vanilla"})
        teacher_indices = torch.tensor([[0, 1]], dtype=torch.long)
        loss_ctx = loss_cfg.build(
            data={
                "shifted_labels": torch.tensor([[-100, 10]], dtype=torch.long),
                "advantages": torch.ones(1, 2),
                "teacher_indices": teacher_indices,
            }
        )

        assert loss_ctx is not None
        torch.testing.assert_close(loss_ctx.loss_kwargs.teacher_indices, teacher_indices)

    def test_distillation_loss_kwargs_split_teacher_indices_with_ignore_padding(self) -> None:
        loss_kwargs = DistillationLossKwargs(
            shifted_labels=torch.ones(1, 2, dtype=torch.long),
            advantages=torch.ones(1, 2),
            teacher_logprobs=torch.ones(1, 2),
            target_token_ids=torch.ones(1, 2, dtype=torch.long),
            teacher_indices=torch.tensor([[0, 1]], dtype=torch.long),
        )
        sp_mesh = SimpleNamespace(size=lambda: 2)

        with (
            patch("xtuner.v1.loss.ce_loss.sp_split", side_effect=lambda tensor, **_: tensor),
            patch("xtuner.v1.rl.loss.base_loss.sp_split", side_effect=lambda tensor, **_: tensor),
            patch(
                "xtuner.v1.rl.loss.distillation_loss.sp_split",
                side_effect=lambda tensor, **_: tensor,
            ) as split,
        ):
            loss_kwargs.sp_split(sp_mesh)

        padding_values = [call.kwargs["padding_value"] for call in split.call_args_list]
        self.assertEqual(padding_values, [0.0, 0, -1])

    def test_finalize_train_metrics_handles_policy_and_distillation_keys(self) -> None:
        loss_cfg = DistillationLossConfig(policy_loss_cfg={"loss_type": "vanilla"})
        metrics = loss_cfg.finalize_metrics(
            {
                "reduced_train_policy_ratio_abs_dev_sum": 2.0,
                "reduced_train_policy_kl1_sum": 4.0,
                "reduced_train_policy_kl3_sum": 6.0,
                "reduced_train_policy_valid_count": 2.0,
                "reduced_train_policy_ratio_max": 1.5,
                "reduced_train_policy_ratio_min": 0.5,
                "reduced_distillation_kl_sum": 8.0,
                "reduced_distillation_abs_loss_sum": 10.0,
                "reduced_distillation_valid_count": 2.0,
                "reduced_opd_reverse_kl_sum": 8.0,
                "reduced_opd_abs_logprob_loss_sum": 10.0,
            },
            "cpu",
        )

        self.assertEqual(metrics["reduced_train_policy_ratio_abs_dev_mean"], 1.0)
        self.assertEqual(metrics["reduced_train_policy_kl1"], 2.0)
        self.assertEqual(metrics["reduced_train_policy_kl3"], 3.0)
        self.assertEqual(metrics["reduced_train_policy_ratio_max"], 1.5)
        self.assertEqual(metrics["reduced_train_policy_ratio_min"], 0.5)
        self.assertEqual(metrics["reduced_distillation_kl"], 4.0)
        self.assertEqual(metrics["reduced_distillation_abs_loss"], 5.0)
        self.assertEqual(metrics["opd_reverse_kl"], 4.0)
        self.assertEqual(metrics["opd_abs_logprob_loss"], 5.0)


class TestRolloutTeacherClient(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _response(payload: dict) -> httpx.Response:
        return httpx.Response(
            200,
            request=httpx.Request("POST", "http://teacher/generate"),
            json=payload,
        )

    def _build_topk_client(self, *, max_retry_per_sample: int = 0) -> RolloutTeacherClient:
        target_config = TeacherTargetConfig(
            mode="topk",
            top_k=2,
        )
        with patch.dict(
            "os.environ",
            {"XTUNER_USE_LMDEPLOY": "1", "XTUNER_USE_SGLANG": "0", "XTUNER_USE_VLLM": "0"},
        ):
            client = RolloutTeacherClient(
                RolloutTeacherConfig(
                    name="teacher",
                    endpoints=["http://teacher"],
                    max_retry_per_sample=max_retry_per_sample,
                ),
                target_config,
            )
        self.addAsyncCleanup(client._client.aclose)
        return client

    @staticmethod
    def _state() -> RolloutState:
        return RolloutState(
            group_id=1,
            message=[],
            prompt_ids=[10, 11, 12],
            response_ids=[13, 14],
            status=Status.COMPLETED,
            extra_fields={"origin_data_source": "math"},
        )

    async def test_compute_sampled_token_logprobs_uses_current_interface(self) -> None:
        response = httpx.Response(
            200,
            request=httpx.Request("POST", "http://teacher/generate"),
            json={
                "meta_info": {
                    "prompt_tokens": 5,
                    "input_token_logprobs": [
                        [-0.1, 11],
                        [-0.2, 12],
                        [-0.3, 13],
                        [-0.4, 14],
                    ],
                }
            },
        )
        target_config = TeacherTargetConfig(mode="sampled_token")
        with patch.dict(
            "os.environ",
            {"XTUNER_USE_LMDEPLOY": "1", "XTUNER_USE_SGLANG": "0", "XTUNER_USE_VLLM": "0"},
        ):
            client = RolloutTeacherClient(
                RolloutTeacherConfig(name="teacher", endpoints=["http://teacher"]),
                target_config,
            )
        self.addAsyncCleanup(client._client.aclose)
        client._client.post = AsyncMock(return_value=response)
        state = RolloutState(
            group_id=1,
            message=[],
            prompt_ids=[10, 11, 12],
            response_ids=[13, 14],
            status=Status.COMPLETED,
            extra_fields={"origin_data_source": "math"},
        )

        result = await client.compute_logprobs(state)

        self.assertEqual(result.status, Status.COMPLETED)
        self.assertEqual(result.teacher_targets.kind, "sampled")
        self.assertEqual(result.teacher_targets.tokens, [13, 14])
        self.assertEqual(result.teacher_targets.logprobs, [-0.3, -0.4])
        self.assertIn("teacher_score_time_s", result.extra_fields)

    async def test_malformed_sampled_response_becomes_failed_state(self) -> None:
        response = self._response(
            {
                "meta_info": {
                    "input_token_logprobs": [
                        [-0.1, 11],
                        [-0.2, 12],
                        [-0.3, 13],
                        [-0.4, 14],
                    ]
                }
            }
        )
        target_config = TeacherTargetConfig(mode="sampled_token")
        with patch.dict(
            "os.environ",
            {"XTUNER_USE_LMDEPLOY": "1", "XTUNER_USE_SGLANG": "0", "XTUNER_USE_VLLM": "0"},
        ):
            client = RolloutTeacherClient(
                RolloutTeacherConfig(
                    name="teacher",
                    endpoints=["http://teacher"],
                    max_retry_per_sample=0,
                ),
                target_config,
            )
        self.addAsyncCleanup(client._client.aclose)
        client._client.post = AsyncMock(return_value=response)

        result = await client.compute_logprobs(self._state())

        self.assertEqual(result.status, Status.FAILED)
        self.assertIn("prompt_tokens", result.error_msg or "")

    async def test_sampled_response_rejects_non_numeric_logprob(self) -> None:
        response = self._response(
            {
                "meta_info": {
                    "prompt_tokens": 5,
                    "input_token_logprobs": [
                        [-0.1, 11],
                        [-0.2, 12],
                        [True, 13],
                        [-0.4, 14],
                    ],
                }
            }
        )
        target_config = TeacherTargetConfig(mode="sampled_token")
        with patch.dict(
            "os.environ",
            {"XTUNER_USE_LMDEPLOY": "1", "XTUNER_USE_SGLANG": "0", "XTUNER_USE_VLLM": "0"},
        ):
            client = RolloutTeacherClient(
                RolloutTeacherConfig(
                    name="teacher",
                    endpoints=["http://teacher"],
                    max_retry_per_sample=0,
                ),
                target_config,
            )
        self.addAsyncCleanup(client._client.aclose)
        client._client.post = AsyncMock(return_value=response)

        result = await client.compute_logprobs(self._state())

        self.assertEqual(result.status, Status.FAILED)
        self.assertIn("non-numeric logprob", result.error_msg or "")

    async def test_malformed_topk_responses_become_failed_states(self) -> None:
        valid_rows = [
            [[-0.1, 1], [-0.2, 2]],
            [[-0.3, 3], [-0.4, 4]],
            [[-0.5, 5], [-0.6, 6]],
            [[-0.7, 7], [-0.8, 8]],
        ]
        malformed_payloads = {
            "missing_meta_info": {},
            "missing_topk_field": {"meta_info": {"prompt_tokens": 5}},
            "wrong_topk_type": {"meta_info": {"prompt_tokens": 5, "input_top_logprobs": "invalid"}},
            "wrong_row_count": {"meta_info": {"prompt_tokens": 5, "input_top_logprobs": valid_rows[:-1]}},
            "ragged_k": {"meta_info": {"prompt_tokens": 5, "input_top_logprobs": [*valid_rows[:-1], [[-0.7, 7]]]}},
            "invalid_token_id": {
                "meta_info": {
                    "prompt_tokens": 5,
                    "input_top_logprobs": [*valid_rows[:-1], [[-0.7, "7"], [-0.8, 8]]],
                }
            },
        }

        for case_name, payload in malformed_payloads.items():
            with self.subTest(case=case_name):
                client = self._build_topk_client()
                client._client.post = AsyncMock(return_value=self._response(payload))

                result = await client.compute_logprobs(self._state())

                self.assertEqual(result.status, Status.FAILED)
                self.assertIsNone(result.teacher_targets)
                self.assertIn("last_error=", result.error_msg or "")
                client._client.post.assert_awaited_once()

        client = self._build_topk_client()
        non_finite_response = httpx.Response(
            200,
            request=httpx.Request("POST", "http://teacher/generate"),
            headers={"content-type": "application/json"},
            content=(
                b'{"meta_info":{"prompt_tokens":5,"input_top_logprobs":'
                b"[[[-0.1,1],[-0.2,2]],[[-0.3,3],[-0.4,4]],"
                b"[[0.5,5],[-0.6,6]],[[NaN,7],[-0.8,8]]]}}"
            ),
        )
        client._client.post = AsyncMock(return_value=non_finite_response)

        result = await client.compute_logprobs(self._state())

        self.assertEqual(result.status, Status.FAILED)
        self.assertIn("NaN or Inf", result.error_msg or "")

    async def test_invalid_topk_response_is_retried_before_success(self) -> None:
        invalid_response = self._response({"meta_info": {"prompt_tokens": 5}})
        valid_response = self._response(
            {
                "meta_info": {
                    "prompt_tokens": 5,
                    "input_top_logprobs": [
                        [[-0.1, 1], [-0.2, 2]],
                        [[-0.3, 3], [-0.4, 4]],
                        [[-0.5, 5], [-0.6, 6]],
                        [[-0.7, 7], [-0.8, 8]],
                    ],
                }
            }
        )
        client = self._build_topk_client(max_retry_per_sample=1)
        client._client.post = AsyncMock(side_effect=[invalid_response, valid_response])

        with patch("xtuner.v1.rl.distillation.rollout_teacher_manager.asyncio.sleep", new=AsyncMock()):
            result = await client.compute_logprobs(self._state())

        self.assertEqual(result.status, Status.COMPLETED)
        self.assertEqual(result.teacher_targets.kind, "topk")
        self.assertEqual(result.teacher_targets.tokens, [[5, 6], [7, 8]])
        self.assertEqual(result.teacher_targets.logprobs, [[-0.5, -0.6], [-0.7, -0.8]])
        self.assertEqual(client._client.post.await_count, 2)
        request_payload = client._client.post.await_args.kwargs["json"]
        self.assertEqual(request_payload["input_ids"], [10, 11, 12, 13, 14])
        self.assertEqual(request_payload["top_logprobs_num"], 2)


class TestTopKTrainingController(unittest.TestCase):
    def test_packs_targets_along_sequence_dimension(self) -> None:
        controller = TrainingController(workers=[])
        first = {
            "seq_ctx": SequenceContext.from_input_ids((torch.tensor([[1, 2]]),), device="cpu"),
            "shifted_labels": torch.tensor([[-100, 2]]),
            "advantage": [0.0, 0.0],
            "rollout_logprobs": torch.zeros(1, 2),
            "teacher_logprobs": torch.tensor([[[-0.1, -0.2], [-0.3, -0.4]]]),
            "target_token_ids": torch.tensor([[[1, 2], [3, 4]]]),
        }
        second = {
            "seq_ctx": SequenceContext.from_input_ids((torch.tensor([[3]]),), device="cpu"),
            "shifted_labels": torch.tensor([[3]]),
            "advantage": [0.0],
            "rollout_logprobs": torch.zeros(1, 1),
            "teacher_logprobs": torch.tensor([[[-0.5, -0.6]]]),
            "target_token_ids": torch.tensor([[[5, 6]]]),
        }

        packed = controller._packing([first, second], pack_max_length=4, language_cfg=None)

        self.assertEqual(len(packed), 1)
        self.assertEqual(packed[0]["teacher_logprobs"].shape, (1, 4, 2))
        self.assertEqual(packed[0]["target_token_ids"].shape, (1, 4, 2))
        torch.testing.assert_close(packed[0]["teacher_logprobs"][0, 3], torch.zeros(2))
        torch.testing.assert_close(packed[0]["target_token_ids"][0, 3], torch.zeros(2, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
