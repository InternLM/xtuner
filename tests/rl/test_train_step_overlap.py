"""Train-step orchestration overlap: CPU preparation, background trajectory
dumping and non-blocking rollout offload around ``_train_one_batch``.

The colocate step overlaps three independent phases between rollout and the
NPU update:

1. trajectory dumping (reads the rollout states, writes JSONL) moves to a
   single-worker background thread;
2. the rollout-engine offload is issued non-blocking;
3. the training controller's CPU preparation (reward clustering, pack plan,
   object-store staging) runs while (1) and (2) are in flight.

The NPU phases only start after the trajectory dump joined (fail-before-fit)
and the offload completed.
"""

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.trainer.controller import TrainingController
from xtuner.v1.rl.trainer.pack import PackPlan, RankPlan
from xtuner.v1.train.rl_trainer import RLColocateTrainer


def _trainable_state(uid=1):
    state = RolloutState(
        rollout_id=uid,
        group_id=uid,
        message=[{"role": "user", "content": "prompt"}],
        input_ids=[10, 11, 20, 21],
        labels=[-100, -100, 20, 21],
        response="answer",
        status=Status.COMPLETED,
        reward={"score": 1.0},
    )
    state.set_training_fields(input_ids=state.input_ids, labels=state.labels, logprobs=None)
    return state


def _pack_plan():
    return PackPlan(ranks=(RankPlan(global_indices=(0,), steps=(((0,),),)),), metrics={})


class TestPrepareFitSplit(unittest.TestCase):
    """``prepare_fit`` runs the CPU phases only; ``session.fit`` the NPU ones."""

    def _controller(self):
        controller = TrainingController.__new__(TrainingController)
        controller.workers = [
            SimpleNamespace(
                get_worker_cfg=SimpleNamespace(remote=MagicMock(return_value={"pack_max_length": 16})),
                get_data_replicate_size=SimpleNamespace(remote=MagicMock(return_value=1)),
                set_residency=SimpleNamespace(remote=MagicMock(return_value=None)),
                actor_fit=SimpleNamespace(remote=MagicMock(return_value={"grad_norm": 0.0})),
            )
        ]
        controller._has_critic = False
        controller.logger = MagicMock()
        return controller

    def _prepared(self, state):
        return {
            "rollout_items": [state],
            "reward_groups": [],
            "batch_attr": {},
            "cluster_rewards": [1.0],
            "distillation_reward_observations": [],
        }

    def test_prepare_fit_does_not_touch_workers_or_load_batch(self):
        controller = self._controller()
        state = _trainable_state()
        with (
            patch.object(
                TrainingController, "_prepare_rollout_items", return_value=self._prepared(state)
            ) as prepare_items,
            patch.object(TrainingController, "_build_pack_plan", return_value=_pack_plan()) as build_plan,
            patch.object(TrainingController, "_loaded_batch") as loaded_batch,
            patch(
                "xtuner.v1.rl.trainer.controller.ray.get", side_effect=lambda value, timeout=None: value
            ) as ray_get,
        ):
            session = controller.prepare_fit([[_trainable_state()]], pack_max_length=16, rollout_idx=3)

        prepare_items.assert_called_once()
        self.assertEqual(build_plan.call_args[0][3], 1)  # data_replicate_size from the worker
        self.assertEqual(ray_get.call_count, 2)  # worker cfg + replicate size, no NPU call
        loaded_batch.assert_not_called()
        controller.workers[0].actor_fit.remote.assert_not_called()
        self.assertIs(session._pack_plan, build_plan.return_value)
        self.assertIs(session._prepared["rollout_items"][0], state)

    def test_session_fit_runs_load_and_actor_update_once(self):
        controller = self._controller()
        with (
            patch.object(
                TrainingController, "_prepare_rollout_items", return_value=self._prepared(_trainable_state())
            ),
            patch.object(TrainingController, "_build_pack_plan", return_value=_pack_plan()),
            patch.object(TrainingController, "_loaded_batch") as loaded_batch,
            patch.object(TrainingController, "_prepare_group_targets", return_value=[1.0]) as group_targets,
            patch.object(TrainingController, "_build_trainer_log_info", return_value={"rewards/mean": 1.0}),
            patch.object(TrainingController, "_critic_log_scalars", return_value={}),
            patch(
                "xtuner.v1.rl.trainer.controller.ray.get", side_effect=lambda value, timeout=None: value
            ),
        ):
            session = controller.prepare_fit([[_trainable_state()]], pack_max_length=16, rollout_idx=3)
            log_infos, data_info = session.fit(raw_rewards_sum=2.0, raw_rewards_count=2)

        loaded_batch.assert_called_once()
        group_targets.assert_called_once()
        controller.workers[0].actor_fit.remote.assert_called_once_with()
        controller.workers[0].set_residency.remote.assert_called_once_with("actor", "actor")
        self.assertEqual(log_infos, [{"grad_norm": 0.0}])
        self.assertEqual(data_info["rewards/mean"], 1.0)

    def test_fit_wrapper_equivalent_to_prepare_then_execute(self):
        controller = self._controller()
        with (
            patch.object(
                TrainingController, "_prepare_rollout_items", return_value=self._prepared(_trainable_state())
            ),
            patch.object(TrainingController, "_build_pack_plan", return_value=_pack_plan()),
            patch.object(TrainingController, "_loaded_batch"),
            patch.object(TrainingController, "_prepare_group_targets", return_value=[1.0]),
            patch.object(TrainingController, "_build_trainer_log_info") as log_info,
            patch.object(TrainingController, "_critic_log_scalars", return_value={}),
            patch(
                "xtuner.v1.rl.trainer.controller.ray.get", side_effect=lambda value, timeout=None: value
            ),
        ):
            controller.fit([[_trainable_state()]], pack_max_length=16, rollout_idx=5, raw_rewards_sum=3.0)

        self.assertEqual(log_info.call_args.kwargs["raw_rewards_sum"], 3.0)


class TestTrainOneBatchOverlap(unittest.TestCase):
    """``_train_one_batch`` overlaps dump/offload/prep and joins before fit."""

    def _make_trainer(self, tmpdir):
        trainer = RLColocateTrainer.__new__(RLColocateTrainer)
        trainer.logger = MagicMock()
        trainer._train_worker_cfg = SimpleNamespace(pack_max_length=16)
        trainer._meta = SimpleNamespace(latest_exp=SimpleNamespace(exp_dir=str(Path(tmpdir) / "exp")))
        trainer._trajectory_save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-traj")
        trainer._prepare_train_batch = MagicMock(return_value=(0, []))
        trainer._release_train_batch_resources = MagicMock()
        trainer._train_nccl_suspended = False
        return trainer

    def test_overlap_ordering_and_fail_before_fit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer(tmpdir)
            events = []
            # The save thread parks until the CPU preparation started; a
            # synchronous save would deadlock on this handshake instead of
            # overlapping, so the wait doubles as the overlap proof.
            prepare_started = threading.Event()

            def save_trajectories(batch, path):
                events.append("save_start")
                if not prepare_started.wait(timeout=5.0):
                    raise TimeoutError("trajectory save must overlap CPU preparation")
                events.append("save_done")

            trainer._save_trajectories = MagicMock(side_effect=save_trajectories)

            trainer.rollout_controller = SimpleNamespace(
                offload=SimpleNamespace(
                    remote=MagicMock(side_effect=lambda: events.append("offload_issued") or "ref")
                )
            )
            fit = MagicMock(side_effect=lambda **kwargs: events.append("fit"))

            def prepare_fit(*args, **kwargs):
                events.append("prepare")
                prepare_started.set()
                return SimpleNamespace(fit=fit)

            trainer.train_controller = SimpleNamespace(prepare_fit=MagicMock(side_effect=prepare_fit))

            with patch(
                "xtuner.v1.train.rl_trainer.ray.get",
                side_effect=lambda obj, timeout=None: events.append("offload_joined") or obj,
            ):
                trainer._train_one_batch(
                    [[_trainable_state()]],
                    train_step=1,
                    step_timer_dict={},
                    offload_rollout_before_train=True,
                )

            # offload is issued non-blocking before the CPU preparation; the
            # trajectory dump joins first, then the offload, then fit runs.
            self.assertIn("save_start", events)
            self.assertLess(events.index("offload_issued"), events.index("prepare"))
            self.assertLess(events.index("prepare"), events.index("save_done"))
            self.assertLess(events.index("save_done"), events.index("offload_joined"))
            self.assertEqual(events[-1], "fit")
            trajectory_path = Path(trainer.exp_dir) / "train_rollout" / "train_rollout_1.jsonl"
            trainer._save_trajectories.assert_called_once()
            self.assertEqual(trainer._save_trajectories.call_args[0][1], trajectory_path)
            trainer._release_train_batch_resources.assert_called_once()

    def test_trajectory_save_failure_fails_before_fit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer(tmpdir)

            def boom(batch, path):
                raise OSError("disk full")

            trainer._save_trajectories = MagicMock(side_effect=boom)
            fit = MagicMock()
            trainer.train_controller = SimpleNamespace(
                prepare_fit=MagicMock(return_value=SimpleNamespace(fit=fit))
            )
            trainer.rollout_controller = SimpleNamespace(offload=SimpleNamespace(remote=MagicMock()))

            with patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj, timeout=None: obj):
                with self.assertRaisesRegex(OSError, "disk full"):
                    trainer._train_one_batch(
                        [[_trainable_state()]],
                        train_step=1,
                        step_timer_dict={},
                        offload_rollout_before_train=True,
                    )

            fit.assert_not_called()
            trainer._release_train_batch_resources.assert_not_called()

    def test_no_offload_flag_skips_rollout_offload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer(tmpdir)
            trainer._save_trajectories = MagicMock()
            fit = MagicMock(return_value=([{"train_metrics": []}], {}))
            trainer.train_controller = SimpleNamespace(
                prepare_fit=MagicMock(return_value=SimpleNamespace(fit=fit))
            )
            offload = SimpleNamespace(remote=MagicMock())
            trainer.rollout_controller = SimpleNamespace(offload=offload)

            with patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj, timeout=None: obj):
                trainer._train_one_batch([[_trainable_state()]], train_step=1, step_timer_dict={})

            offload.remote.assert_not_called()
            fit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
