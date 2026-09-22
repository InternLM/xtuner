"""RLDataPacker index-only packing plan 的 contract 测试。

Packer 只产出 ``[dp][optimizer_step][pack][sample_index]`` 索引计划与 padding 统计，
不触碰样本数据。测试覆盖设计约束：每个 index 恰好出现一次、pack 不超长、DP/step 维度
pack 数一致、optimizer steps 多于 pack 数、data-replica 分组、deterministic seed、
padding 统计，以及 legacy 策略与重构前 controller greedy pack + interleaved DP 分配 +
worker ``iters_per_step`` 重分组的精确等价性。
"""

import math
import unittest

from xtuner.v1.rl.trainer.pack import RLDataPacker


def _legacy_reference_plan(
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

    def test_legacy_matches_previous_schedule(self):
        # lengths 刻意制造多 pack、非整除与 DP 填充场景。
        lengths = [10, 8, 6, 30, 12, 4, 7, 20, 2, 9]
        packer = self._make_packer(world_size=4, optimizer_steps=2)

        plan, padding = packer.pack(lengths)

        self.assertEqual(plan, _legacy_reference_plan(lengths, self.PACK_MAX_LENGTH, dp_size=4, optimizer_steps=2))
        self._assert_plan_invariants(plan, lengths, optimizer_steps=2)
        total_packs = sum(len(packs) for rank in plan for packs in rank)
        self.assertEqual(padding, total_packs * self.PACK_MAX_LENGTH - sum(lengths))

    def test_legacy_regroups_when_packs_fewer_than_optimizer_steps(self):
        # packs_per_dp=2 < optimizer_steps=4：旧 worker iters_per_step=ceil(2/4)=1，
        # 实际只训练 2 个 step。
        lengths = [16, 16, 16, 16]
        packer = self._make_packer(world_size=2, optimizer_steps=4)

        plan, padding = packer.pack(lengths)

        self.assertEqual(plan, _legacy_reference_plan(lengths, self.PACK_MAX_LENGTH, dp_size=2, optimizer_steps=4))
        self.assertEqual(len(plan[0]), 2)
        self.assertEqual(padding, 4 * 2 * self.PACK_MAX_LENGTH - sum(lengths))

    def test_single_pack_interleaves_across_dp(self):
        # 全部样本拼进一个 pack 时，interleaved 分配把唯一 pack 给 rank0，其余 rank 全空 pack。
        lengths = [8, 4]
        packer = self._make_packer(world_size=4, optimizer_steps=1)

        plan, padding = packer.pack(lengths)

        self.assertEqual(plan, _legacy_reference_plan(lengths, self.PACK_MAX_LENGTH, dp_size=4, optimizer_steps=1))
        self.assertEqual(plan[0][0], [[0, 1]])
        self.assertEqual(plan[1][0], [[]])
        self.assertEqual(padding, 4 * self.PACK_MAX_LENGTH - sum(lengths))

    def test_data_replicate_size_divides_world(self):
        # world=8、replicate=2 → dp=4，各 replica group 共享同一 plan。
        lengths = [10, 10, 10, 10, 10, 10]
        packer = self._make_packer(world_size=8, data_replicate_size=2, optimizer_steps=1)

        plan, _ = packer.pack(lengths)

        self.assertEqual(len(plan), 4)
        self.assertEqual(plan, _legacy_reference_plan(lengths, self.PACK_MAX_LENGTH, dp_size=4, optimizer_steps=1))

    def test_oversize_sample_raises(self):
        packer = self._make_packer()

        with self.assertRaisesRegex(ValueError, "exceeds pack_max_length"):
            packer.pack([self.PACK_MAX_LENGTH + 1])

    def test_empty_lengths(self):
        packer = self._make_packer()

        plan, padding = packer.pack([])

        self.assertEqual(plan, [])
        self.assertEqual(padding, 0)

    def test_unknown_strategy_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown packing strategy"):
            self._make_packer(pack_strategy="unknown")

    def test_greedy_strategy_invariants(self):
        lengths = [10, 8, 6, 30, 12, 4, 7, 20, 2, 9]
        packer = self._make_packer(pack_strategy="greedy")

        plan, padding = packer.pack(lengths)

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


if __name__ == "__main__":
    unittest.main()
