import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.agent_loop_manager.agent_loop_manager import AgentLoopManager, _TaskRunner
from xtuner.v1.rl.agent_loop_manager.producer import SyncProduceStrategyConfig
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig


class _FakeSampler:
    def __init__(self):
        self.saved_paths: list[Path] = []
        self.resumed_paths: list[Path] = []

    def __len__(self):
        return 1

    def save(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        (checkpoint_path / "dataloader").write_text("fake", encoding="utf-8")
        self.saved_paths.append(checkpoint_path)

    def resume(self, checkpoint_path):
        self.resumed_paths.append(Path(checkpoint_path))


def _build_manager():
    replay_buffer = SyncReplayBufferConfig().build()
    task_runner = _TaskRunner(
        task_name="unit_task",
        agent_loop=SimpleNamespace(),
        produce_strategy=SyncProduceStrategyConfig().build(),
        sampler=_FakeSampler(),
    )
    return AgentLoopManager(task_runners=[task_runner], replay_buffer=replay_buffer)


def _rollout_group(index: int) -> list[RolloutState]:
    return [
        RolloutState(
            uid=index,
            message_uid=index,
            message=[{"role": "user", "content": "question"}],
            prompt_ids=[1, 2, 3],
            response="ok",
            response_ids=[100],
            reward={"score": 1.0},
            status=Status.COMPLETED,
            extra_fields={"index": index},
        )
    ]


class TestAgentLoopManagerCheckpointFast(unittest.TestCase):
    def test_save_without_replay_buffer_skips_buffer_file_and_resumes_empty_buffer(self):
        async def _run():
            manager = _build_manager()
            progress = manager._produce_progress
            progress.next_consumer_step = 3
            progress.producer_future_step = 5
            progress.consumed_samples["unit_task"] = 2
            progress.target_samples["unit_task"] = 5
            progress.target_upto_future_step = 5
            progress.raw_rewards_sum["unit_task"] = 4.0
            progress.raw_rewards_count["unit_task"] = 4
            progress.produced_samples["unit_task"] = 3
            progress.produced_tokens["unit_task"] = 12
            progress.produce_time_s = 1.5
            await manager.replay_buffer.put(_rollout_group(7), "unit_task", model_step=2, current_train_step=3)

            with tempfile.TemporaryDirectory() as tmp_dir:
                checkpoint_path = Path(tmp_dir) / "ckpt"
                await manager.save(checkpoint_path, model_step=2, no_save_replay_buffer=True)

                self.assertFalse((checkpoint_path / "replay_buffer.pth").exists())
                with (checkpoint_path / "agent_loop_manager_state.json").open("r", encoding="utf-8") as f:
                    manager_state = json.load(f)
                self.assertFalse(manager_state["replay_buffer_saved"])
                self.assertEqual(manager_state["producer_future_step"], 3)
                self.assertEqual(manager_state["target_upto_future_step"], 2)
                self.assertEqual(manager_state["target_samples"], {"unit_task": 2})
                self.assertEqual(manager_state["raw_rewards_sum"], {"unit_task": 0.0})
                self.assertEqual(manager_state["produced_samples"], {"unit_task": 0})

                restored_manager = _build_manager()
                restored_model_step = await restored_manager.resume(checkpoint_path)

            self.assertEqual(restored_model_step, 2)
            self.assertEqual(len(restored_manager.replay_buffer), 0)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
