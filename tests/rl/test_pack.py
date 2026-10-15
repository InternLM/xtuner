"""RLDataPacker index-only packing plan 的 contract 测试。

Packer 只产出 ``[dp][optimizer_step][pack][sample_index]`` 索引计划与 padding 统计，
不触碰样本数据。测试覆盖设计约束：每个 index 恰好出现一次、pack 不超长、DP/step 维度
pack 数一致、optimizer steps 多于 pack 数、data-replica 分组、deterministic seed、
padding 统计，以及 legacy 策略与重构前 controller greedy pack + interleaved DP 分配 +
worker ``iters_per_step`` 重分组的精确等价性。
"""

import heapq
import math
import unittest
from unittest.mock import patch

from xtuner.v1.rl.trainer.pack import RLDataPacker


def _greedy_reference_plan(
    lengths: list[int],
    pack_max_length: int,
    dp_size: int,
    optimizer_steps: int,
) -> list[list[list[list[int]]]]:
    """重构前行为的参考实现：controller greedy pack + DP-multiple 空 pack 填充 +
    interleaved ``[dp::dp]`` 分配 + worker 顺序 ``iters_per_step`` 重分组。"""
    packs: list[list[int]] = []
    current: list[int] = []
    current_len = 0
    for index, length in enumerate(lengths):
        if current_len + length <= pack_max_length:
            current.append(index)
            current_len += length
        else:
            if current:
                packs.append(current)
            current = [index]
            current_len = length
    if current:
        packs.append(current)

    pad_num = math.ceil(len(packs) / dp_size) * dp_size - len(packs)
    packs.extend([[] for _ in range(pad_num)])

    packs_per_dp = len(packs) // dp_size
    iters_per_step = max(1, math.ceil(packs_per_dp / optimizer_steps))
    actual_steps = math.ceil(packs_per_dp / iters_per_step)
    return [
        [packs[dp_rank::dp_size][step * iters_per_step : (step + 1) * iters_per_step] for step in range(actual_steps)]
        for dp_rank in range(dp_size)
    ]


def _lpt_reference_plan(
    lengths: list[int],
    pack_max_length: int,
    dp_size: int,
    optimizer_steps: int,
) -> list[list[list[list[int]]]]:
    """Baseline PPO packer: longest-first into the least-loaded pack, then stride across DP."""
    min_bins = optimizer_steps * dp_size
    cap_bins = math.ceil(math.ceil(sum(lengths) / pack_max_length) / dp_size) * dp_size
    num_bins = max(min_bins, cap_bins)
    order = sorted(range(len(lengths)), key=lambda index: lengths[index], reverse=True)
    while True:
        heap = [(0, bin_index) for bin_index in range(num_bins)]
        heapq.heapify(heap)
        bins: list[list[int]] = [[] for _ in range(num_bins)]
        feasible = True
        for index in order:
            load, bin_index = heap[0]
            if load + lengths[index] <= pack_max_length:
                heapq.heapreplace(heap, (load + lengths[index], bin_index))
                bins[bin_index].append(index)
            else:
                feasible = False
                break
        if feasible:
            break
        num_bins += dp_size

    plan: list[list[list[list[int]]]] = []
    for dp_rank in range(dp_size):
        rank_packs = bins[dp_rank::dp_size]
        num_updates = min(len(rank_packs), optimizer_steps)
        quotient, remainder = divmod(len(rank_packs), num_updates)
        chunks: list[list[list[int]]] = []
        offset = 0
        for chunk_index in range(num_updates):
            chunk_size = quotient + int(chunk_index < remainder)
            chunks.append(rank_packs[offset : offset + chunk_size])
            offset += chunk_size
        plan.append(chunks)
    return plan


class TestRLDataPacker(unittest.TestCase):
    PACK_MAX_LENGTH = 32

    def _make_packer(
        self,
        world_size: int = 4,
        data_replicate_size: int = 1,
        optimizer_steps: int = 2,
        pack_strategy: str = "legacy",
        pack_seed: int | None = None,
    ) -> RLDataPacker:
        return RLDataPacker(
            pack_max_length=self.PACK_MAX_LENGTH,
            world_size=world_size,
            data_replicate_size=data_replicate_size,
            optimizer_steps=optimizer_steps,
            pack_strategy=pack_strategy,  # type: ignore[arg-type]
            pack_seed=pack_seed,
        )

    def _assert_plan_invariants(
        self,
        plan: list[list[list[list[int]]]],
        lengths: list[int],
        optimizer_steps: int,
    ) -> None:
        all_packs = [pack for rank in plan for step in rank for pack in step]
        seen = [index for pack in all_packs for index in pack]
        self.assertEqual(sorted(seen), list(range(len(lengths))))

        for rank in plan:
            self.assertGreaterEqual(len(rank), 1)
            self.assertLessEqual(len(rank), optimizer_steps)
            for packs in rank:
                for pack in packs:
                    self.assertLessEqual(sum(lengths[index] for index in pack), self.PACK_MAX_LENGTH)

        steps_per_rank = {len(rank) for rank in plan}
        self.assertEqual(len(steps_per_rank), 1)
        # 同一 step 中所有 DP rank 的 pack 数一致（按 step 对齐比较）。
        num_steps = next(iter(steps_per_rank))
        for step in range(num_steps):
            counts = {len(rank[step]) for rank in plan}
            self.assertEqual(len(counts), 1)

    @patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", True)
    def test_legacy_matches_previous_schedule(self):
        # lengths 刻意制造多 pack、非整除与 DP 填充场景。
        lengths = [10, 8, 6, 30, 12, 4, 7, 20, 2, 9]
        packer = self._make_packer(world_size=4, optimizer_steps=2)

        plan, padding = packer.pack(lengths)

        self.assertEqual(plan, _greedy_reference_plan(lengths, self.PACK_MAX_LENGTH, dp_size=4, optimizer_steps=2))
        self._assert_plan_invariants(plan, lengths, optimizer_steps=2)
        total_packs = sum(len(packs) for rank in plan for packs in rank)
        self.assertEqual(padding, total_packs * self.PACK_MAX_LENGTH - sum(lengths))

    @patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", True)
    def test_legacy_regroups_when_packs_fewer_than_optimizer_steps(self):
        # packs_per_dp=2 < optimizer_steps=4：旧 worker iters_per_step=ceil(2/4)=1，
        # 实际只训练 2 个 step。
        lengths = [32, 32, 32, 32]
        packer = self._make_packer(world_size=2, optimizer_steps=4)

        plan, padding = packer.pack(lengths)

        self.assertEqual(plan, _greedy_reference_plan(lengths, self.PACK_MAX_LENGTH, dp_size=2, optimizer_steps=4))
        self.assertEqual(len(plan[0]), 2)
        self.assertEqual(padding, 4 * self.PACK_MAX_LENGTH - sum(lengths))

    @patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", True)
    def test_single_pack_interleaves_across_dp(self):
        # 全部样本拼进一个 pack 时，interleaved 分配把唯一 pack 给 rank0，其余 rank 全空 pack。
        lengths = [8, 4]
        packer = self._make_packer(world_size=4, optimizer_steps=1)

        plan, padding = packer.pack(lengths)

        self.assertEqual(plan, _greedy_reference_plan(lengths, self.PACK_MAX_LENGTH, dp_size=4, optimizer_steps=1))
        self.assertEqual(plan[0][0], [[0, 1]])
        self.assertEqual(plan[1][0], [[]])
        self.assertEqual(padding, 4 * self.PACK_MAX_LENGTH - sum(lengths))

    @patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", True)
    def test_data_replicate_size_divides_world(self):
        # world=8、replicate=2 → dp=4，各 replica group 共享同一 plan。
        lengths = [10, 10, 10, 10, 10, 10]
        packer = self._make_packer(world_size=8, data_replicate_size=2, optimizer_steps=1)

        plan, _ = packer.pack(lengths)

        self.assertEqual(len(plan), 4)
        self.assertEqual(plan, _greedy_reference_plan(lengths, self.PACK_MAX_LENGTH, dp_size=4, optimizer_steps=1))

    def test_oversize_sample_raises(self):
        packer = self._make_packer()

        with self.assertRaisesRegex(ValueError, "exceeds pack_max_length"):
            packer.pack([self.PACK_MAX_LENGTH + 1])

    def test_empty_lengths(self):
        packer = self._make_packer()

        plan, padding = packer.pack([])

        self.assertEqual(plan, [])
        self.assertEqual(padding, 0)

    def test_shuffled_indices_use_matching_lengths(self):
        lengths = [2, 8, 8]
        for strategy in ("legacy", "greedy"):
            with self.subTest(strategy=strategy):
                packer = RLDataPacker(10, 2, 1, 2, pack_strategy=strategy)
                with (
                    patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", False),
                    patch(
                        "xtuner.v1.rl.trainer.pack.random.shuffle",
                        side_effect=lambda indices: indices.__setitem__(slice(None), [1, 2, 0]),
                    ),
                ):
                    plan, padding = packer.pack(lengths)
                packs = [pack for rank in plan for step in rank for pack in step]
                self.assertEqual(sorted(index for pack in packs for index in pack), [0, 1, 2])
                self.assertTrue(all(sum(lengths[i] for i in pack) <= 10 for pack in packs))
                self.assertEqual(padding, len(packs) * 10 - sum(lengths))
                self.assertEqual(lengths, [2, 8, 8])
                self.assertEqual(len({len(rank) for rank in plan}), 1)
                for step in range(len(plan[0])):
                    self.assertEqual(len({len(rank[step]) for rank in plan}), 1)

    def test_greedy_fills_gaps_from_sft_candidate_buffer(self):
        lengths = [6, 7, 4]
        with patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", True):
            legacy, old_padding = RLDataPacker(10, 1, 1, 1, pack_strategy="legacy").pack(lengths)
            greedy, new_padding = RLDataPacker(10, 1, 1, 1, pack_strategy="greedy").pack(lengths)
        self.assertEqual(legacy, [[[[0], [1], [2]]]])
        self.assertEqual(greedy, [[[[0, 2], [1]]]])
        self.assertEqual((old_padding, new_padding), (13, 3))

    def test_zero_buffer_keeps_sequential_pack_order(self):
        lengths = [6, 7, 4]
        with patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", True):
            legacy = RLDataPacker(10, 1, 1, 1, pack_strategy="legacy").pack(lengths)
            greedy = RLDataPacker(10, 1, 1, 1, pack_strategy="greedy", pack_extra_buffer_size=0).pack(lengths)
        self.assertEqual(legacy, greedy)

    def test_unknown_strategy_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown packing strategy"):
            self._make_packer(pack_strategy="unknown")

    def test_legacy_matches_main_across_replica_and_step_layouts(self):
        # main controller: packs[dp_rank::dp_size]; worker: ceil(packs_per_dp / steps).
        for dp_size in (1, 2, 4):
            for replicas in (1, 2):
                for steps in (1, 2, 4):
                    for count in (1, 7, 17, 33):
                        lengths = [32] * count
                        with self.subTest(dp_size=dp_size, replicas=replicas, steps=steps, count=count):
                            packer = self._make_packer(
                                world_size=dp_size * replicas,
                                data_replicate_size=replicas,
                                optimizer_steps=steps,
                            )
                            with patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", True):
                                plan, _ = packer.pack(lengths)
                            self.assertEqual(plan, _greedy_reference_plan(lengths, 32, dp_size, steps))

    @patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", True)
    def test_greedy_interleaves_instead_of_contiguous_dp_blocks(self):
        lengths = [32] * 12
        packer = self._make_packer(pack_strategy="greedy")

        plan, padding = packer.pack(lengths)

        self.assertEqual(plan, [[[[rank], [rank + 4]], [[rank + 8]]] for rank in range(4)])
        self._assert_plan_invariants(plan, lengths, optimizer_steps=2)
        total_packs = sum(len(packs) for rank in plan for packs in rank)
        self.assertEqual(padding, total_packs * self.PACK_MAX_LENGTH - sum(lengths))

    def test_balance_strategy_is_deterministic_and_invariants(self):
        lengths = [10, 8, 6, 30, 12, 4, 7, 20, 2, 9, 15, 3]
        first = self._make_packer(optimizer_steps=2, pack_strategy="balance", pack_seed=7)
        second = self._make_packer(optimizer_steps=2, pack_strategy="balance", pack_seed=7)

        plan, padding = first.pack(lengths)
        plan_again, _ = second.pack(lengths)

        self.assertEqual(plan, plan_again)
        self._assert_plan_invariants(plan, lengths, optimizer_steps=2)
        total_packs = sum(len(packs) for rank in plan for packs in rank)
        self.assertEqual(padding, total_packs * self.PACK_MAX_LENGTH - sum(lengths))

    def test_native_strategy_invariants(self):
        lengths = [10, 8, 6, 30, 12, 4, 7]
        packer = self._make_packer(world_size=4, optimizer_steps=1, pack_strategy="native")

        plan, padding = packer.pack(lengths)

        self._assert_plan_invariants(plan, lengths, optimizer_steps=1)
        total_packs = sum(len(packs) for rank in plan for packs in rank)
        self.assertEqual(padding, total_packs * self.PACK_MAX_LENGTH - sum(lengths))

    def test_native_negative_slots_become_empty_packs(self):
        # 7 个样本分 4 个 dp：负数 pad 槽从计划中移除后表现为空 pack。
        lengths = [8, 8, 8, 8, 8, 8, 8]
        packer = self._make_packer(world_size=4, optimizer_steps=1, pack_strategy="native")

        plan, padding = packer.pack(lengths)

        seen = [index for rank in plan for step in rank for pack in step for index in pack]
        self.assertEqual(sorted(seen), list(range(7)))
        self.assertGreaterEqual(padding, 4 * self.PACK_MAX_LENGTH - sum(lengths))

    def test_lpt_matches_baseline_schedule(self):
        # 30 进 bin0，20 进 bin1，两条 10 分别进 bin2/bin3，8 补进当时最空的 bin2，4 补进 bin3。
        lengths = [30, 20, 10, 10, 8, 4]
        packer = self._make_packer(world_size=2, optimizer_steps=2, pack_strategy="lpt")

        with patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", True):
            plan, padding = packer.pack(lengths)

        expected = [
            [[[0]], [[2, 4]]],
            [[[1]], [[3, 5]]],
        ]
        self.assertEqual(plan, expected)
        self.assertEqual(plan, _lpt_reference_plan(lengths, self.PACK_MAX_LENGTH, dp_size=2, optimizer_steps=2))
        self._assert_plan_invariants(plan, lengths, optimizer_steps=2)
        total_packs = sum(len(packs) for rank in plan for packs in rank)
        self.assertEqual(padding, total_packs * self.PACK_MAX_LENGTH - sum(lengths))

    def test_lpt_grows_bins_and_keeps_empty_pack(self):
        # 两条 20 占满初始的 2 个 bin 后，第三条放不下，bin 数加到 4，留下一个空 pack。
        lengths = [20, 20, 20]
        packer = self._make_packer(world_size=2, optimizer_steps=1, pack_strategy="lpt")

        with patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", True):
            plan, padding = packer.pack(lengths)

        self.assertEqual(plan, [[[[0], [2]]], [[[1], []]]])
        self.assertEqual(plan, _lpt_reference_plan(lengths, self.PACK_MAX_LENGTH, dp_size=2, optimizer_steps=1))
        self.assertEqual(padding, 4 * self.PACK_MAX_LENGTH - sum(lengths))

    def test_lpt_gives_remainder_packs_to_earlier_steps(self):
        # 6 个 pack、2 个 step：前面的 step 多拿 1 个 pack。
        lengths = [16] * 10
        packer = self._make_packer(world_size=2, optimizer_steps=2, pack_strategy="lpt")

        with patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", True):
            plan, padding = packer.pack(lengths)

        self.assertEqual(plan, _lpt_reference_plan(lengths, self.PACK_MAX_LENGTH, dp_size=2, optimizer_steps=2))
        self.assertEqual([len(step) for step in plan[0]], [2, 1])
        self.assertEqual([len(step) for step in plan[1]], [2, 1])
        self._assert_plan_invariants(plan, lengths, optimizer_steps=2)
        total_packs = sum(len(packs) for rank in plan for packs in rank)
        self.assertEqual(padding, total_packs * self.PACK_MAX_LENGTH - sum(lengths))

    def test_lpt_shuffles_finished_packs(self):
        lengths = [30, 20, 10, 10, 8, 4]
        packer = self._make_packer(world_size=2, optimizer_steps=2, pack_strategy="lpt")

        def reverse_packs(packs):
            packs.reverse()

        with (
            patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", False),
            patch("xtuner.v1.rl.trainer.pack.random.shuffle", reverse_packs),
        ):
            plan, _ = packer.pack(lengths)

        # 未打乱时是 [[0], [1], [2, 4], [3, 5]]；反转后 stride 到两个 rank。
        self.assertEqual(plan, [[[[3, 5]], [[1]]], [[[2, 4]], [[0]]]])


if __name__ == "__main__":
    unittest.main()
