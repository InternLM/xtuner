import random
import unittest
from types import SimpleNamespace

import torch

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.rl.trainer.pack import RLDataPacker
from xtuner.v1.rl.trainer.worker import TrainingWorker


class TestDataBatchPacker(unittest.TestCase):
    def setUp(self):
        self.pack_max_length = 3072

    def _run_strategy_test(
        self,
        strategy,
        world_size,
        optimizer_steps,
        lengths,
        pack_max_length,
        expected_padding=None,
    ):
        packer = RLDataPacker(
            pack_max_length=pack_max_length,
            world_size=world_size,
            data_replicate_size=1,
            optimizer_steps=optimizer_steps,
            pack_strategy=strategy,
        )

        packed_indices, padding_tokens = packer.pack(lengths)

        all_packs = [
            pack_indices
            for rank_indices in packed_indices
            for step_indices in rank_indices
            for pack_indices in step_indices
        ]
        seen_indices = [data_index for pack_indices in all_packs for data_index in pack_indices]
        self.assertEqual(sorted(seen_indices), list(range(len(lengths))))
        pack_token_counts = [sum(lengths[data_index] for data_index in pack_indices) for pack_indices in all_packs]
        self.assertTrue(all(pack_tokens <= pack_max_length for pack_tokens in pack_token_counts))
        self.assertEqual(len(all_packs) * pack_max_length, sum(lengths) + padding_tokens)

        if strategy == "balance":
            rank_token_counts = [
                sum(
                    lengths[data_index]
                    for step_indices in rank_indices
                    for pack_indices in step_indices
                    for data_index in pack_indices
                )
                for rank_indices in packed_indices
            ]
            self.assertLessEqual(max(rank_token_counts) - min(rank_token_counts), max(lengths))

        if expected_padding is not None:
            self.assertEqual(padding_tokens, expected_padding)

    def test_variable_packs(self):
        lengths = [1500, 1000, 2800, 3000, 1500, 2000, 2100, 1000, 800]
        self._run_strategy_test("native", 2, 2, lengths, self.pack_max_length, 15020)
        self._run_strategy_test("balance", 2, 2, lengths, self.pack_max_length, 8876)
        self._run_strategy_test("greedy", 2, 2, lengths, self.pack_max_length, 8876)

    def test_imbalance_dp_size(self):
        lengths = [500]
        for strategy in ["native", "balance", "greedy"]:
            self._run_strategy_test(strategy, 2, 1, lengths, self.pack_max_length, 5644)

    def test_imbalanced_steps(self):
        lengths = [100, 200, 2500, 3000, 50, 400, 1000, 1500]
        self._run_strategy_test("native", 2, 4, lengths, self.pack_max_length, 15826)
        self._run_strategy_test("balance", 2, 4, lengths, self.pack_max_length, 15826)
        self._run_strategy_test("greedy", 2, 4, lengths, self.pack_max_length, 3538)

    def test_random_lengths(self):
        lengths = [random.randint(1, 32768) for _ in range(1024)]
        for strategy in ["native", "balance", "greedy"]:
            self._run_strategy_test(strategy, 8, 16, lengths, 32768)

    def test_native_supports_pack_length_below_split_size(self):
        self._run_strategy_test("native", 2, 1, [128], 256, 384)


class TestTrainingWorkerPackMaterialization(unittest.TestCase):
    @staticmethod
    def _create_dummy_item(length: int, value: int):
        input_ids = torch.full((1, length), value, dtype=torch.long)
        seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cpu")
        return {
            "seq_ctx": seq_ctx,
            "shifted_labels": torch.full((1, length), value, dtype=torch.long),
            "advantages": torch.full((1, length), float(value), dtype=torch.float32),
            "rollout_logprobs": torch.full((1, length), float(value), dtype=torch.float32),
        }

    def test_worker_selects_indices_and_materializes_packs(self):
        worker = TrainingWorker.__new__(TrainingWorker)
        worker.config = SimpleNamespace(pack_max_length=8, model_cfg=None)
        data_batches = [self._create_dummy_item(3, 1), self._create_dummy_item(2, 2)]

        packed_data = worker._materialize_packs(data_batches, [[[0, 1]], [[]]])

        self.assertEqual(len(packed_data), 2)
        self.assertEqual(packed_data[0][0]["seq_ctx"].input_ids.numel(), 8)
        self.assertEqual(packed_data[0][0]["seq_ctx"].num_padding, 3)
        self.assertEqual(packed_data[1][0]["seq_ctx"].num_padding, 8)
        self.assertEqual(packed_data[0][0]["seq_ctx"].input_ids[0, :5].tolist(), [1, 1, 1, 2, 2])


if __name__ == "__main__":
    unittest.main()
