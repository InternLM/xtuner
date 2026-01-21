import unittest
import torch
import math
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.rl.base.pack import DataBatchPacker

class TestDataBatchPacker(unittest.TestCase):
    def setUp(self):
        self.pack_max_length = 3072
        self.split_size = 1024

    def _create_dummy_item(self, length: int, val=1):
        """构造带特定值的样本,用于验证解包一致性"""
        input_ids = torch.full((1, length), val, dtype=torch.long)
        cu_seq_lens_q = torch.tensor([0, length], dtype=torch.int32)
        cu_seq_lens_k = torch.tensor([0, length], dtype=torch.int32)
        max_length_q = torch.tensor(length, dtype=torch.int32)
        max_length_k = torch.tensor(length, dtype=torch.int32)
        seq_ctx = SequenceContext(
            input_ids=input_ids,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
            num_padding=0,
            device="cpu",
        )
        return {
            "seq_ctx": seq_ctx,
            "shifted_labels": torch.full((1, length), val, dtype=torch.long),
            "advantages": torch.full((1, length), float(val), dtype=torch.float),
            "rollout_logprobs": torch.full((1, length), float(val), dtype=torch.float),
        }

    def _run_strategy_test(self, strategy, world_size, optimizer_steps, lengths, expected_padding):
        data_batches = [self._create_dummy_item(l, val=7) for l in lengths]
        total_data_tokens = sum(lengths)
        
        packer = DataBatchPacker(
            pack_max_length=self.pack_max_length,
            world_size=world_size,
            data_replicate_size=1,
            optimizer_steps=optimizer_steps,
            pack_strategy=strategy
        )
        
        packed_res, padding_tokens = packer.pack(data_batches)
        
        self.assertEqual(padding_tokens, expected_padding, 
                         f"Strategy {strategy} padding mismatch. Expected {expected_padding}, got {padding_tokens}")
        
        all_packs = []
        for rank_data in packed_res:
            for step_data in rank_data:
                all_packs.extend(step_data)
        
        total_capacity = len(all_packs) * self.pack_max_length
        self.assertEqual(total_capacity, total_data_tokens + padding_tokens)

        valid_token_count = sum((p["seq_ctx"].input_ids != 0).sum().item() for p in all_packs)
        valid_label_count = sum((p["shifted_labels"] != -100).sum().item() for p in all_packs)
        valid_adv_count = sum((p["advantages"] != -100).sum().item() for p in all_packs)
        
        self.assertEqual(valid_token_count, total_data_tokens)
        self.assertEqual(valid_label_count, total_data_tokens)
        self.assertEqual(valid_adv_count, total_data_tokens)

    def test_case1_not_enough_for_dp(self):
        """Case 1: 1条数据(500), dp=2, steps=1。
        - balance_split 将 1 条补为 2 条 (添加 3072 token 的补丁项) -> 3072
        - Rank 0 (500) 打包补齐 -> 3072 - 500 = 2572
        - Rank 1 (1024) 打包补齐 -> 3072 - 3072 = 0
        - total = 3072 + 2572 = 5644
        """
        packer = DataBatchPacker(
            pack_max_length=self.pack_max_length, world_size=2,
            data_replicate_size=1, optimizer_steps=1, pack_strategy="balance"
        )
        data_batches = [self._create_dummy_item(500)]
        packed_res, padding_tokens = packer.pack(data_batches)
        
        self.assertEqual(len(packed_res), 2)  # 2 Ranks
        self.assertEqual(padding_tokens, 5644)
        all_packs = []
        for rank_data in packed_res:
            for step_data in rank_data:
                all_packs.extend(step_data)
        valid_token_count = sum((p["seq_ctx"].input_ids != 0).sum().item() for p in all_packs)
        valid_label_count = sum((p["shifted_labels"] != -100).sum().item() for p in all_packs)
        # import debugpy
        # debugpy.connect(("0.0.0.0", 5678))
        # debugpy.breakpoint()
        valid_adv_count = sum((p["advantages"] != -100).sum().item() for p in all_packs)
        self.assertEqual(valid_token_count, 500)
        self.assertEqual(valid_label_count, 500)
        self.assertEqual(valid_adv_count, 500)


    def test_case1_imbalance_dp_size(self):
        """Case 1: 1条短数据(500), dp=2, steps=1
        说明：所有策略最终都会补齐到 2 个 Rank 的 Pack 深度。
        预期 Padding: 3072 * 2 - 500 = 5644
        """
        lengths = [500]
        for strat in ["native", "balance", "greedy"]:
            self._run_strategy_test(strat, 2, 1, lengths, 5644)

    def test_case2_imbalanced_steps(self):
        """Case 2: 8条数据,dp=2, steps=4
        总 tokens: 8750
        - Native/Balance: 每个 Step 一个 Pack,共 2 rank * 4 steps * 1 pack = 8 packs.
          Padding: 8 * 3072 - 8850 = 15726
        - Greedy: 采用贪心合并,这 8 条可以打成很紧凑的 3 个 Pack。
          DP=2 对齐后变为 4 个 Pack (每卡2个 Pack)。
          Greedy 代码会发现 each_dp_batches_num=2 < optimizer_steps=4,于是 actual_steps 变成 2。
          最终容量: 2 rank * 2 steps * 1 pack = 4 packs.
          Padding: 4 * 3072 - 8850 = 3438
        """
        lengths = [100, 200, 2500, 3000, 50, 400, 1000, 1500]
        self._run_strategy_test("native", 2, 4, lengths, 15826)
        self._run_strategy_test("balance", 2, 4, lengths, 15826)
        self._run_strategy_test("greedy", 2, 4, lengths, 3538)

    def test_case3_variable_packs(self):
        """Case 3: 利用特定长度触发 Pack 数不一致, dp=2, steps=2
        总 tokens: 12700
        - Native: 
            1. DP Rank 切分： 
                rank0: [1500, 1000, 2800, 1500]
                rank1: [2000, 2100, 1000, 800]
            2. Optimizer steps切分：
                rank0: [1500, 1000], [2800, 1500]
                rank1: [2000, 2100], [1000, 800]
            3 pack and padding
                rank0: step0: [2500 -> 3072],                 step1: [2800 -> 3072], [1500 -> 3072], 
                rank1: step0: [2100 -> 3072], [2000 -> 3072], step1: [1800 -> 3072]
            4. 跨卡对齐pack数量： 
                rank0: step0: [2500 -> 3072], [0 -> 3072]     step1: [2800 -> 3072], [1500 -> 3072], 
                rank1: step0: [2100 -> 3072], [2000 -> 3072], step1: [1800 -> 3072], [0 -> 3072]
            padding_tokens: 3072 - 2500 + 3072 + 3072 - 2800 + 3072 - 1500 + 3072 - 2100 + 3072 - 2000 + 3072 - 1800 + 3072 = 11876
        - Balance: 
            1. DP Rank 均衡切分：
                rank0: [2800, 1500, 1000, 1000]
                rank1: [2000, 2100, 1500, 800]
            2. optimizer_steps 均衡切分：
                rank0: [2800], [1500, 1000, 1000]
                rank1: [2100, 800], [2000, 1500]
            3. pack and padding: 
                rank0: step0: [2800 -> 3072],  step1: [2500 -> 3072], [1000 -> 3072], 
                rank1: step0: [2900 -> 3072],  step1: [2000 -> 3072], [1500 -> 3072]
            4. 跨卡对齐pack数量：
                skip
            padding_tokens: 3072 - 2800 + 3072 - 2500 + 3072 - 1000 + 3072 - 2900 + 3072 - 2000 + 3072 - 1500 = 5732
        - Greedy (贪心优先打包): 追求 Pack 填充率最大化,不维护样本在 Step 间的顺序
            1. pack and padding:
               Pack 1: [1500, 1000] -> [2500 -> 3072]
               Pack 2: [2800]       -> [2800 -> 3072]
               Pack 3: [1500]       -> [1500 -> 3072]
               Pack 4: [2000]       -> [2000 -> 3072]
               Pack 5: [2100]       -> [2100 -> 3072]
               Pack 6: [1000, 800]  -> [1800 -> 3072]
            2. DP 切分:
               rank0: [Pack 1, Pack 2, Pack 3]
               rank1: [Pack 4, Pack 5, Pack 6]
            3. Opitmizer steps 切分:
               rank0: step0: [Pack 1, Pack 2], step1: [Pack 3]
               rank1: step0: [Pack 4, Pack 5], step1: [Pack 6]
            4. 跨卡对齐pack数量：
               skip
            padding_tokens: 3072 - 2500 + 3072 - 2800 + 3072 - 1500 + 3072 - 2000 + 3072 - 2100 + 3072 - 1800 = 5732
        """
        lengths = [1500, 1000, 2800, 1500, 2000, 2100, 1000, 800]
        self._run_strategy_test("native", 2, 2, lengths, 11876)
        self._run_strategy_test("balance", 2, 2, lengths, 5732)
        self._run_strategy_test("greedy", 2, 2, lengths, 5732)

if __name__ == "__main__":
    unittest.main()