import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import torch

from recipe.on_policy_distillation.build_teacher_server_commands import (
    build_teacher_launch_server_commands,
)
from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.model import Qwen3Dense4BConfig
from xtuner.v1.rl.distillation import (
    DistillationConfig,
    DistillationTrainerAdapter,
    RolloutTeacherClient,
    RolloutTeacherConfig,
    TeacherTargetConfig,
    TrainTeacherConfig,
    TrainTeacherManager,
    TrainTeacherManagerConfig,
    TrainTeacherOutputs,
    TrainTeacherTimings,
)
from xtuner.v1.rl.distillation import train_teacher_manager as train_teacher_manager_module
from xtuner.v1.rl.loss import DistillationLossConfig, DistillationLossKwargs
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
            rollout_config=SimpleNamespace(cross_weight_update_rollout=None),
        )

        with patch("xtuner.v1.train.rl_trainer.AutoTokenizer.from_pretrained", return_value=MagicMock()):
            trainer._build_agent_loop_components(config, replay_buffer=MagicMock())

        self.assertIs(trainer.agent_loop_manager, built_manager)
        self.assertIsNone(manager_config.build.call_args.kwargs["rollout_teacher_scorer_config"])

    def test_train_teacher_prefetch_is_projected_to_manager_config(self) -> None:
        loss_config = DistillationLossConfig(policy_loss_cfg={"loss_type": "vanilla"})
        distillation_config = DistillationConfig(
            loss_config=loss_config,
            teachers=[
                TrainTeacherConfig(
                    name="teacher",
                    model_path="/models/teacher",
                    model_cfg=Qwen3Dense4BConfig(),
                    residency="disk",
                )
            ],
            data_source_teacher_map={"math": "teacher"},
            train_teacher_prefetch=False,
        )

        manager_config = distillation_config.train_teacher_manager_config

        self.assertIsNotNone(manager_config)
        assert manager_config is not None
        self.assertFalse(manager_config.prefetch)

    def test_disk_teacher_ignores_fsdp_cpu_offload(self) -> None:
        teacher_config = TrainTeacherConfig(
            name="teacher",
            model_path="/models/teacher",
            model_cfg=Qwen3Dense4BConfig(),
            fsdp_cfg=FSDPConfig(cpu_offload=True),
            residency="disk",
        )
        manager = TrainTeacherManager.__new__(TrainTeacherManager)
        manager._teacher_configs = [teacher_config]
        manager._teacher_process_groups = [[]]

        with patch(
            "xtuner.v1.rl.distillation.train_teacher_manager.build_frozen_model",
            return_value=MagicMock(),
        ) as build:
            manager._build_disk_teacher(0, "cpu")

        self.assertTrue(teacher_config.fsdp_cfg.cpu_offload)
        self.assertFalse(build.call_args.args[2].cpu_offload)
        self.assertEqual(build.call_args.kwargs, {"target_device": "cpu", "load_device": "cpu"})


class TestTrainTeacherTimings(unittest.TestCase):
    def test_process_group_registry_is_hidden_behind_guarded_seam(self) -> None:
        first_group = object()
        second_group = object()
        world = SimpleNamespace(pg_map={first_group: object(), second_group: object()})

        with (
            patch.object(train_teacher_manager_module.dist, "is_available", return_value=True),
            patch.object(train_teacher_manager_module.dist, "is_initialized", return_value=True),
            patch.object(train_teacher_manager_module, "_DIST_WORLD", world),
        ):
            groups = train_teacher_manager_module._registered_process_groups()

        self.assertEqual(groups, (first_group, second_group))

    def test_process_group_registry_reports_incompatible_pytorch(self) -> None:
        with (
            patch.object(train_teacher_manager_module.dist, "is_available", return_value=True),
            patch.object(train_teacher_manager_module.dist, "is_initialized", return_value=True),
            patch.object(train_teacher_manager_module, "_DIST_WORLD", None),
            self.assertRaisesRegex(RuntimeError, "process-group registry.*unavailable"),
        ):
            train_teacher_manager_module._registered_process_groups()

    def test_disk_teacher_is_not_built_during_manager_init(self) -> None:
        config = TrainTeacherManagerConfig(
            teachers=[
                TrainTeacherConfig(
                    name="teacher",
                    model_path="/models/teacher",
                    model_cfg=Qwen3Dense4BConfig(),
                    residency="disk",
                )
            ],
            data_source_teacher_map={"math": "teacher"},
            target_config=TeacherTargetConfig(mode="sampled_token"),
        )

        with patch("xtuner.v1.rl.distillation.train_teacher_manager.build_frozen_model") as build:
            manager = TrainTeacherManager(config, chunk_size=None)

        self.assertEqual(manager._teachers, [None])
        build.assert_not_called()

    def test_disk_teacher_prefetch_can_be_disabled(self) -> None:
        manager = TrainTeacherManager.__new__(TrainTeacherManager)
        manager._teacher_configs = [SimpleNamespace(residency="disk")]
        manager._prefetch_enabled = False
        manager._prefetch_executor = MagicMock()
        manager._prefetch_future = None

        manager._start_teacher_prefetch(0)

        manager._prefetch_executor.submit.assert_not_called()
        self.assertIsNone(manager._prefetch_future)

    def test_prefetch_prepares_cpu_tensors_for_main_thread_model_shell(self) -> None:
        manager = TrainTeacherManager.__new__(TrainTeacherManager)
        teacher_config = SimpleNamespace(
            residency="disk", model_cfg=MagicMock(), model_path="/model", fsdp_cfg=None
        )
        manager._teacher_configs = [teacher_config]
        manager._teacher_names = ["teacher"]
        manager._teacher_process_groups = [[]]
        manager._prefetch_enabled = True
        manager._prefetch_executor = MagicMock()
        manager._prefetch_future = None
        model = MagicMock()

        with (
            patch(
                "xtuner.v1.rl.distillation.train_teacher_manager.build_frozen_model_shell",
                return_value=model,
            ) as build_shell,
            patch("xtuner.v1.rl.distillation.train_teacher_manager.prepare_frozen_model") as prepare,
        ):
            manager._start_teacher_prefetch(0)

        build_shell.assert_called_once_with(teacher_config.model_cfg, None)
        manager._prefetch_executor.submit.assert_called_once_with(prepare, model, teacher_config.model_path)

    def test_lifecycle_timings_are_recorded_for_every_teacher(self) -> None:
        manager = TrainTeacherManager.__new__(TrainTeacherManager)
        manager._teacher_configs = [
            SimpleNamespace(name="teacher", residency="cpu"),
            SimpleNamespace(name="teacher", residency="cpu"),
        ]
        first_teacher = MagicMock()
        second_teacher = MagicMock()
        manager._teachers = [first_teacher, second_teacher]
        timings = TrainTeacherTimings()

        with (
            patch(
                "xtuner.v1.rl.distillation.train_teacher_manager.time.perf_counter",
                side_effect=[1.0, 2.0, 3.0, 5.0, 6.0, 9.0, 10.0, 14.0, 15.0, 20.0, 21.0, 27.0],
            ),
            patch.object(manager, "_synchronize_device"),
            patch.object(manager, "_offload_to_cpu"),
        ):
            with manager._teacher_on_device(0, timings, "teacher_a"):
                pass
            with manager._teacher_on_device(1, timings, "teacher_b"):
                pass

        self.assertEqual(
            timings.to_dict(),
            {
                "teacher_a": {"compute": 2.0, "onload": 1.0, "offload": 3.0},
                "teacher_b": {"compute": 5.0, "onload": 4.0, "offload": 6.0},
            },
        )
        self.assertEqual((timings.compute, timings.onload, timings.offload), (7.0, 5.0, 9.0))

    def test_disk_teacher_is_rebuilt_and_dropped_for_each_use(self) -> None:
        manager = TrainTeacherManager.__new__(TrainTeacherManager)
        manager._teacher_configs = [
            SimpleNamespace(
                name="teacher", residency="disk", model_cfg=MagicMock(), model_path="/models/teacher", fsdp_cfg=None
            )
        ]
        manager._teachers = [None]
        manager._teacher_process_groups = [[]]
        manager._prefetch_executor = None
        manager._prefetch_future = None
        manager._prefetch_teacher_index = None
        first_teacher = MagicMock()
        second_teacher = MagicMock()

        with (
            patch(
                "xtuner.v1.rl.distillation.train_teacher_manager.build_frozen_model",
                side_effect=[first_teacher, second_teacher],
            ) as build,
            patch.object(manager, "_synchronize_device"),
            patch.object(manager, "_empty_device_cache") as empty_cache,
        ):
            with manager._teacher_on_device(0, TrainTeacherTimings(), "teacher") as teacher:
                self.assertIs(teacher, first_teacher)
            self.assertIsNone(manager._teachers[0])
            with manager._teacher_on_device(0, TrainTeacherTimings(), "teacher") as teacher:
                self.assertIs(teacher, second_teacher)
            self.assertIsNone(manager._teachers[0])

        self.assertEqual(build.call_count, 2)
        self.assertEqual(empty_cache.call_count, 2)
        first_teacher.to_device.assert_called_once()
        second_teacher.to_device.assert_called_once()

    def test_disk_teacher_destroys_only_process_groups_created_by_its_build(self) -> None:
        manager = TrainTeacherManager.__new__(TrainTeacherManager)
        manager._teacher_configs = [
            SimpleNamespace(
                name="teacher", residency="disk", model_cfg=MagicMock(), model_path="/models/teacher", fsdp_cfg=None
            )
        ]
        manager._teachers = [None]
        manager._teacher_process_groups = [[]]
        manager._prefetch_executor = None
        manager._prefetch_future = None
        manager._prefetch_teacher_index = None
        existing_group = object()
        teacher_group = object()

        with (
            patch(
                "xtuner.v1.rl.distillation.train_teacher_manager.build_frozen_model",
                return_value=MagicMock(),
            ),
            patch.object(
                manager,
                "_snapshot_process_groups",
                side_effect=[(existing_group,), (existing_group, teacher_group)],
            ),
            patch.object(manager, "_destroy_process_groups") as destroy,
            patch.object(manager, "_synchronize_device"),
            patch.object(manager, "_empty_device_cache"),
        ):
            with manager._teacher_on_device(0, TrainTeacherTimings(), "teacher"):
                pass

        destroy.assert_called_once_with([teacher_group])
        self.assertEqual(manager._teacher_process_groups, [[]])

    def test_disk_teacher_consumes_cpu_prefetch_and_starts_next(self) -> None:
        manager = TrainTeacherManager.__new__(TrainTeacherManager)
        manager._teacher_configs = [
            SimpleNamespace(
                name="teacher", residency="disk", model_cfg=MagicMock(), model_path="/models/teacher", fsdp_cfg=None
            )
        ]
        manager._teachers = [None]
        manager._teacher_names = ["teacher"]
        manager._teacher_process_groups = [[]]
        prefetched_teacher = MagicMock()
        future: Future[tuple[MagicMock, MagicMock]] = Future()
        future.set_result((MagicMock(), MagicMock()))
        manager._prefetch_executor = MagicMock()
        manager._prefetch_future = future
        manager._prefetch_teacher_index = 0

        with (
            patch("xtuner.v1.rl.distillation.train_teacher_manager.build_frozen_model") as build,
            patch(
                "xtuner.v1.rl.distillation.train_teacher_manager.materialize_frozen_model",
                return_value=prefetched_teacher,
            ),
            patch.object(manager, "_start_teacher_prefetch") as start_prefetch,
            patch.object(manager, "_synchronize_device"),
            patch.object(manager, "_empty_device_cache"),
        ):
            with manager._teacher_on_device(0, TrainTeacherTimings(), "teacher") as teacher:
                self.assertIs(teacher, prefetched_teacher)
                start_prefetch.assert_called_once_with(1)

        build.assert_not_called()
        prefetched_teacher.to_device.assert_called_once()
        self.assertIsNone(manager._teachers[0])
        self.assertIsNone(manager._prefetch_future)
        self.assertIsNone(manager._prefetch_teacher_index)

    def test_disk_teacher_is_dropped_when_compute_fails(self) -> None:
        manager = TrainTeacherManager.__new__(TrainTeacherManager)
        manager._teacher_configs = [
            SimpleNamespace(
                name="teacher", residency="disk", model_cfg=MagicMock(), model_path="/models/teacher", fsdp_cfg=None
            )
        ]
        manager._teachers = [None]
        manager._teacher_process_groups = [[]]
        manager._prefetch_executor = None
        manager._prefetch_future = None
        manager._prefetch_teacher_index = None

        with (
            patch(
                "xtuner.v1.rl.distillation.train_teacher_manager.build_frozen_model",
                return_value=MagicMock(),
            ),
            patch.object(manager, "_synchronize_device"),
            patch.object(manager, "_empty_device_cache") as empty_cache,
        ):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                with manager._teacher_on_device(0, TrainTeacherTimings(), "teacher"):
                    raise RuntimeError("forward failed")

        self.assertIsNone(manager._teachers[0])
        empty_cache.assert_called_once_with()

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
        torch.testing.assert_close(
            loss_ctx.loss_kwargs.teacher_indices,
            teacher_indices.to(loss_ctx.loss_kwargs.teacher_indices.device),
        )

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

    def test_distillation_loss_finalizes_policy_and_distillation_keys(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
