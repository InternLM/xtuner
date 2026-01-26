import unittest
import torch
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.rl.base.pack import RLDataPacker

class TestDataBatchPacker(unittest.TestCase):
    def setUp(self):
        self.pack_max_length = 3072
        self.split_size = 1024

    def _create_dummy_item(self, length: int, val=1):
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

    def _run_strategy_test(self, strategy, world_size, optimizer_steps, lengths, pack_max_length, expected_padding = None):
        data_batches = [self._create_dummy_item(l, val=7) for l in lengths]
        total_data_tokens = sum(lengths)
        
        packer = RLDataPacker(
            pack_max_length=pack_max_length,
            world_size=world_size,
            data_replicate_size=1,
            optimizer_steps=optimizer_steps,
            pack_strategy=strategy
        )
        
        packed_res, padding_tokens = packer.pack(data_batches)
        
        # 验证均衡性：理想情况下，balance 策略分配给各卡的 token 总数差异应该小于单个样本的最大长度
        if strategy == "balance":
            rank_token_counts = []
            for  rank_data in packed_res:
                rank_total_valid_tokens = 0
                for step_data in rank_data:
                    for pack in step_data:
                        # 统计非零（非 padding）的有效 token 数量
                        valid_tokens = (pack["seq_ctx"].input_ids != 0).sum().item()
                        rank_total_valid_tokens += valid_tokens
                rank_token_counts.append(rank_total_valid_tokens)
            
            max_tokens = max(rank_token_counts)
            min_tokens = min(rank_token_counts)
            diff = max_tokens - min_tokens
            max_sample_len = max(lengths) if lengths else 0
            self.assertLessEqual(diff, max_sample_len, 
                f"Balance strategy failed: Token distribution is too skewed. "
                f"Rank counts: {rank_token_counts}, Max diff: {diff}")

        # 对于固定输入，验证padding_tokens是否符合预期来验证pack逻辑正确性
        if expected_padding is not None:    
            self.assertEqual(padding_tokens, expected_padding, f"Strategy {strategy} padding mismatch. Expected {expected_padding}, got {padding_tokens}")
            
        all_packs = []
        for rank_data in packed_res:
            for step_data in rank_data:
                for pack in step_data:
                    self.assertEqual(pack["seq_ctx"].input_ids.numel(), pack_max_length, f"Strategy {strategy} pack length mismatch.")
                    all_packs.append(pack)

        # 验证pack前后的总有效token数是否一致
        total_capacity = len(all_packs) * pack_max_length
        self.assertEqual(total_capacity, total_data_tokens + padding_tokens)

        all_input_ids = torch.cat([p["seq_ctx"].input_ids for p in all_packs], dim=1)
        valid_token_count = (all_input_ids != 0).sum().item()
        all_labels = torch.cat([p["shifted_labels"] for p in all_packs], dim=1)
        valid_label_count = (all_labels != -100).sum().item()
        all_advantages = torch.cat([p["advantages"] for p in all_packs], dim=1)
        valid_adv_count = (all_advantages != -100).sum().item()

        self.assertEqual(valid_token_count, total_data_tokens)
        self.assertEqual(valid_label_count, total_data_tokens)
        self.assertEqual(valid_adv_count, total_data_tokens)

    def test_variable_packs(self):
        """随机tokens数输入, dp=2, optimizer_steps=2
        - Native: 
            1. 预处理，保证样本数量能被整除, padding到1024, 这样可以与有效的样本一起Pack
               [1500, 1000, 2800, 3000, 1500, 2000, 2100, 1000, 800] -> padding: [1500, 1000, 2800, 3000, 1500, 2000, 2100, 1000, 800, 1024]
            2. DP Rank 切分： 
                rank0: [1500, 1000, 2800, 3000, 1500]
                rank1: [2000, 2100, 1000,  800, 1024]
            3. Optimizer steps切分：
                rank0: [1500, 1000, 2800], [3000, 1500]
                rank1: [2000, 2100, 1000], [ 800, 1024]
            4 pack and padding
                rank0: step0: [2500 -> 3072], [2800 -> 3072],                 step1: [3000 -> 3072], [1500 -> 3072], 
                rank1: step0: [2000 -> 3072], [2100 -> 3072], [1000 -> 3072], step1: [1824 -> 3072]
            5. 跨卡对齐pack数量： 
                rank0: step0: [2500 -> 3072], [2800 -> 3072], [0 -> 3072]     step1: [3000 -> 3072], [1500 -> 3072], 
                rank1: step0: [2100 -> 3072], [2000 -> 3072], [1000 -> 3072], step1: [1824 -> 3072], [0 -> 3072]
            padding_tokens: 1024 + 3072 - 2500 + 3072 - 2800 + 3072 + 3072 - 3000 + 3072 - 1500 + 3072 - 2100 + 3072 - 2000 + 3072 - 1000 + 3072 - 1824 + 3072 = 15020
        - Balance: 
            1. 对原始输入数据进行排序：
                [1500, 1000, 2800, 3000, 1500, 2000, 2100, 1000, 800] -> [3000, 2800, 2100, 2000, 1500, 1500, 1000, 1000, 800]
            2. 相近长度的N个样本分到N张卡上, 每N个样本为作为N张卡的一次optimizer step的数据
                rank0: [3000, 1500, 800], [2100, 1000],
                rank1: [2800, 1500],      [2000, 1000],
            3. pack and pad:
                rank0: step0: [3000 -> 3072], [2300 -> 3072],   step1: [2100 ->3072], [1000 -> 3072],
                rank1: step0: [2800 -> 3072], [1500 -> 3072],   step1: [3000 ->3072], [.  0 -> 3072],
            4. 跨卡对齐pack数量：
                skip
            padding_tokens: 3072 - 3000 + 3072 - 2300 + 3072 - 2100 + 3072 - 1000 + 3072 - 2800 + 3072 - 1500 + 3072 - 3000 + 3072 = 8876
        - Greedy: 追求 Pack 填充率最大化
            1. pack and padding:
               Pack 1: [1500, 1000] -> [2500 -> 3072]
               Pack 2: [2800]       -> [2800 -> 3072]
               Pack 3: [3000]       -> [3000 -> 3072]
               Pack 4: [1500]       -> [1500 -> 3072]
               Pack 5: [2000]       -> [2000 -> 3072]
               Pack 6: [2100]       -> [2100 -> 3072]
               Pack 7: [1000, 800]  -> [1800 -> 3072]
               Pack 8: [ ]          -> [0    -> 3072] (padding)
            2. DP 切分:
               rank0: [Pack 1, Pack 2, Pack 3, Pack 4]
               rank1: [Pack 5, Pack 6, Pack 7, Pack 8]
            3. Opitmizer steps 切分:
               rank0: step0: [Pack 1, Pack 2], step1: [Pack 3, Pack 4]
               rank1: step0: [Pack 5, Pack 6], step1: [Pack 7, Pack 8]
            4. 跨卡对齐pack数量：
               skip
            padding_tokens: 3072 - 2500 + 3072 - 2800 + 3072 - 3000 + 3072 - 1500 + 3072 - 2000 + 3072 - 2100 + 3072 - 1800 + 3072 = 8876
        """
        lengths = [1500, 1000, 2800, 3000, 1500, 2000, 2100, 1000, 800]
        self._run_strategy_test("native", 2, 2, lengths, self.pack_max_length, 15020)
        self._run_strategy_test("balance", 2, 2, lengths, self.pack_max_length, 8876)
        self._run_strategy_test("greedy", 2, 2, lengths, self.pack_max_length, 8876)

    def test_imbalance_dp_size(self):
        lengths = [500]
        for strat in ["native", "balance", "greedy"]:
            self._run_strategy_test(strat, 2, 1, lengths, self.pack_max_length, 5644)

    def test_imbalanced_steps(self):
        lengths = [100, 200, 2500, 3000, 50, 400, 1000, 1500]
        self._run_strategy_test("native", 2, 4, lengths, self.pack_max_length, 15826)
        self._run_strategy_test("balance", 2, 4, lengths, self.pack_max_length, 15826)
        self._run_strategy_test("greedy", 2, 4, lengths, self.pack_max_length, 3538)

    def test_random_lengths(self):
        import random
        lengths = [random.randint(1, 32768) for _ in range(1024)]
        for strat in ["native", "balance", "greedy"]:
            self._run_strategy_test(strat, 8, 16, lengths, 32768)

if __name__ == "__main__":
    unittest.main()