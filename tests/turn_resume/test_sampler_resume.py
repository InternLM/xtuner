import tempfile
import unittest
from pathlib import Path

import torch

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.datasets.dataloader import Dataloader
from xtuner.v1.datasets.sampler import ParallelSampler
from xtuner.v1.rl.agent_loop_manager.sampler import Sampler


class EpochDataloader:
    def __init__(self):
        self.total = 0
        self.epoch = 0
        self.offset = 0

    def __iter__(self):
        for index in range(self.offset, 8):
            self.total += 1
            yield [RolloutState(message=[{"role": "user", "content": str(index)}])]
        self.offset = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_state_dict(self):
        return {
            "sampler": {"step": self.total % 8, "epoch": self.epoch},
            "total_consumed_samples": self.total,
            "dataset": {},
        }

    def load_state_dict(self, state):
        self.total = state["total_consumed_samples"]
        self.epoch = state["sampler"]["epoch"]
        self.offset = state["sampler"]["step"]


class SamplerResumeTests(unittest.TestCase):
    def test_resume_after_epoch_preserves_suffix_and_unique_ids_for_new_and_legacy_checkpoint(self):
        original = Sampler(EpochDataloader(), 4, None)
        for _ in range(10):
            original.sample_from_dataloader()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            original.save(path)
            for legacy in (False, True):
                with self.subTest(legacy=legacy):
                    state = torch.load(path / "dataloader", weights_only=True)
                    if legacy:
                        state.pop("rollout_group_count", None)
                        state.pop("rollout_next_epoch", None)
                        torch.save(state, path / "dataloader")
                    restored = Sampler(EpochDataloader(), 4, None)
                    restored.resume(path)
                    group = restored.sample_from_dataloader()
                    self.assertEqual([item.group_id for item in group], [10] * 4)
                    self.assertEqual([item.rollout_id for item in group], [40, 41, 42, 43])
                    self.assertEqual(group[0].message[0]["content"], "2")

    def test_real_shuffled_loader_restores_uninterrupted_suffix_at_epoch_boundaries(self):
        def build():
            dataset = [RolloutState(message=[{"role": "user", "content": str(i)}]) for i in range(8)]
            sampler = ParallelSampler(dataset, global_batch_size=1, shuffle=True, seed=42)
            loader = Dataloader(dataset, batch_size=1, sampler=sampler, collate_fn=lambda batch: batch)
            return Sampler(loader, 4, None)

        for consumed in (8, 10, 16):
            for legacy in (False, True):
                with self.subTest(consumed=consumed, legacy=legacy), tempfile.TemporaryDirectory() as tmp:
                    original = build()
                    for _ in range(consumed):
                        original.sample_from_dataloader()
                    original.save(tmp)
                    if legacy:
                        state = torch.load(Path(tmp) / "dataloader", weights_only=True)
                        state.pop("rollout_group_count", None)
                        state.pop("rollout_next_epoch", None)
                        torch.save(state, Path(tmp) / "dataloader")
                    expected = [original.sample_from_dataloader() for _ in range(4)]
                    restored = build()
                    restored.resume(tmp)
                    chained_path = Path(tmp) / "chained"
                    chained_path.mkdir()
                    restored.save(chained_path)
                    chained = build()
                    chained.resume(chained_path)
                    chained_actual = [chained.sample_from_dataloader() for _ in range(4)]
                    self.assertEqual([g[0].message for g in chained_actual], [g[0].message for g in expected])
                    self.assertEqual([g[0].group_id for g in chained_actual], [g[0].group_id for g in expected])
                    actual = [restored.sample_from_dataloader() for _ in range(4)]
                    self.assertEqual([g[0].message for g in actual], [g[0].message for g in expected])
                    self.assertEqual([g[0].group_id for g in actual], [g[0].group_id for g in expected])
