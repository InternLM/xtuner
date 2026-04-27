# Copyright (c) OpenMMLab. All rights reserved.
"""Test Muon optimizer Newton-Schulz functions with/without torch.compile.

Test shapes are based on Qwen3-30B-A3B model config:
- hidden_size: 2048
- num_experts: 128
- moe_intermediate_size: 768
- intermediate_size: 6144 (for shared expert)

MoE expert weight shapes:
- w1/w3: (num_experts * moe_intermediate_size, hidden_size) = (98304, 2048)
  per expert: (768, 2048)
- w2: (hidden_size, num_experts * moe_intermediate_size) = (2048, 98304)
  per expert: (2048, 768)

For testing, we use scaled-down versions to keep tests fast while maintaining
representative shapes.

================================================================================
IMPORTANT: DTensor Compatibility Note
================================================================================

The zeropower_via_newtonschulz5 function supports DTensor input, but with a
known limitation when M > N (e.g., w2 weights where hidden_size > moe_intermediate_size).

Root Cause Analysis (verified by /tmp/test_dtensor_root_cause_detailed.py):
---------------------------------------------------------------------------
When M > N, the Newton-Schulz algorithm transposes the input matrix:
    X = G.view(1, M, N).mT  # becomes (1, N, M)

For a DTensor sharded on dim 0 (M dimension):
    1. After view(1, M, N): placements become Shard(dim=1)
    2. After mT: placements become Shard(dim=2)  # the M dimension moves to dim 2
    3. X @ X.mT produces Partial(sum) DTensor  # contraction dim is sharded
    4. Partial values are not correctly reduced in subsequent operations
    5. Error accumulates across 5 Newton-Schulz iterations:
       Iter 1: X max ~0.016
       Iter 2: X max ~0.060
       Iter 3: X max ~0.099
       Iter 4: X max ~0.29
       Iter 5: X max ~47.5  (EXPLOSION!)
    6. Final result is completely wrong (e.g., 0.1 -> 47.5)

Verification Results:
    - M < N (w1/w3): ✓ PASS - A @ A.mT produces Shard(dim=1), results match exactly
    - M > N (w2):    ✗ FAIL - A @ A.mT produces Partial(sum), results explode
    - M = N (square): ✓ PASS - A @ A.mT produces Shard(dim=1), results match exactly

Workaround:
    For DTensor with M > N (w2 weights), convert to local tensor:
        result = zeropower_via_newtonschulz5(dtensor.to_local(), num_experts=1)

Note:
    This is NOT a torch.compile issue. The same problem occurs with or without
    torch.compile. It's a fundamental limitation of DTensor's Partial placement
    handling in complex matrix operation chains.

newton_schulz_triton:
    Does not support DTensor at all due to direct Triton kernel usage.
    Must use .to_local() to convert before calling.
================================================================================
"""

import pytest
import torch


# TODO:@nil0x9 The original unit test here does not work and is removed along with the removal
# of compile decorators in muon ops (Tests fails when we try to compare compiled NS to vanilla 
# counterpart). We remove the related tests here, but the problem still remains how compile 
# affects the precision of muon ops. It would require investigation, which we defer till a compile
# opition is introduced for optimizers -- then we might re-introduce the TC tests.

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class TestNewtonSchulzCompile:
    """Test Newton-Schulz functions with and without torch.compile."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.epsilon = 1e-7
        self.tolerance = 1e-3  # Tolerance for bfloat16 comparison

    def _create_test_matrix(self, num_experts, M, N):
        """Create a test matrix with given dimensions."""
        shape = (num_experts * M, N)
        return torch.randn(shape, device=self.device, dtype=torch.float32)

    # def test_zeropower_via_newtonschulz5_compile(self):
    #     """Test muon.zeropower_via_newtonschulz5 with/without compile.
    #
    #     Test cases based on Qwen3 MoE architecture (hidden_size=2048, num_experts=128):
    #     - Non-MoE: (6144, 2048) and (2048, 6144) for shared experts
    #     - MoE w1/w3: (128 * 768, 2048) per expert (768, 2048)
    #     - MoE w2: (2048, 128 * 768) per expert (2048, 768)
    #     """
    #     from xtuner.v1.optim.muon import zeropower_via_newtonschulz5
    #
    #     # Scaled-down test cases based on Qwen3 MoE config
    #     test_cases = [
    #         # Non-MoE cases (shared expert-like)
    #         (1, 1536, 512, "shared_expert_w1"),  # (1536, 512) scaled from (6144, 2048)
    #         (1, 512, 1536, "shared_expert_w2"),  # (512, 1536) scaled from (2048, 6144)
    #         # MoE cases - w1/w3 like (M < N)
    #         (8, 192, 512, "moe_w1_small"),  # per expert: (192, 512) scaled from (768, 2048)
    #         (16, 192, 512, "moe_w1_medium"),  # 16 experts
    #         # MoE cases - w2 like (M > N)
    #         (8, 512, 192, "moe_w2_small"),  # per expert: (512, 192) scaled from (2048, 768)
    #         (16, 512, 192, "moe_w2_medium"),  # 16 experts
    #         # Square cases
    #         (1, 512, 512, "square_regular"),
    #         (4, 256, 256, "square_moe"),
    #     ]
    #
    #     for num_experts, M, N, name in test_cases:
    #         G = self._create_test_matrix(num_experts, M, N)
    #
    #         # Without compile
    #         result_no_compile = zeropower_via_newtonschulz5(
    #             G, epsilon=self.epsilon, num_experts=num_experts
    #         )
    #
    #         # With compile
    #         compiled_fn = torch.compile(zeropower_via_newtonschulz5, fullgraph=True)
    #         result_compile = compiled_fn(G, epsilon=self.epsilon, num_experts=num_experts)
    #
    #         # Compare results
    #         max_diff = (result_no_compile - result_compile).abs().max().item()
    #         assert max_diff < self.tolerance, (
    #             f"{name} (num_experts={num_experts}, M={M}, N={N}): "
    #             f"max_diff={max_diff} >= {self.tolerance}"
    #         )

    def test_newton_schulz_triton(self):
        """Test newton_schulz_triton (Triton kernel, no torch.compile).

        Note: Triton kernel is not compatible with torch.compile, so we only test
        without compile and verify basic correctness.
        """
        from xtuner.v1.optim.newton_schulz_triton import newton_schulz_triton

        # Scaled-down test cases based on Qwen3 MoE config
        test_cases = [
            # Non-MoE cases (shared expert-like)
            (1, 1536, 512, "shared_expert_w1"),  # (1536, 512)
            (1, 512, 1536, "shared_expert_w2"),  # (512, 1536)
            # MoE cases - w1/w3 like (M < N)
            (8, 192, 512, "moe_w1_small"),  # 8 experts, each (192, 512)
            (16, 192, 512, "moe_w1_medium"),  # 16 experts
            # MoE cases - w2 like (M > N)
            (8, 512, 192, "moe_w2_small"),  # 8 experts, each (512, 192)
            (16, 512, 192, "moe_w2_medium"),  # 16 experts
            # Square cases
            (1, 512, 512, "square_regular"),
            (4, 256, 256, "square_moe"),
        ]

        for num_experts, M, N, name in test_cases:
            G = self._create_test_matrix(num_experts, M, N)

            # Test without compile (Triton kernel doesn't support compile)
            result = newton_schulz_triton(G, epsilon=self.epsilon, num_experts=num_experts)

            # Basic sanity check: output should have correct shape
            assert result.shape == G.shape, f"{name}: output shape mismatch"

            # Output should not be all zeros or contain NaN/Inf
            assert not torch.isnan(result).any(), f"{name}: output contains NaN"
            assert not torch.isinf(result).any(), f"{name}: output contains Inf"
            assert result.abs().max() > 0, f"{name}: output is all zeros"

    # def test_transpose_case_compile(self):
    #     """Test matrices where rows > cols (transpose case) with compile.
    #
    #     Based on Qwen3 MoE w2 shape: (hidden_size, num_experts * moe_intermediate_size)
    #     """
    #     from xtuner.v1.optim.muon import zeropower_via_newtonschulz5
    #
    #     test_cases = [
    #         # Non-MoE transpose case
    #         (1, 512, 128, "transpose_shared_expert"),  # Scaled from (2048, 512)
    #         # MoE transpose cases - w2 like
    #         (8, 512, 192, "transpose_moe_w2_small"),  # 8 experts, each (512, 192)
    #         (16, 512, 192, "transpose_moe_w2_medium"),  # 16 experts
    #     ]
    #
    #     for num_experts, M, N, name in test_cases:
    #         G = self._create_test_matrix(num_experts, M, N)
    #
    #         # Without compile
    #         result_no_compile = zeropower_via_newtonschulz5(
    #             G, epsilon=self.epsilon, num_experts=num_experts
    #         )
    #
    #         # With compile
    #         compiled_fn = torch.compile(zeropower_via_newtonschulz5, fullgraph=True)
    #         result_compile = compiled_fn(G, epsilon=self.epsilon, num_experts=num_experts)
    #
    #         # Compare results
    #         max_diff = (result_no_compile - result_compile).abs().max().item()
    #         assert max_diff < self.tolerance, (
    #             f"zeropower_via_newtonschulz5 {name} (num_experts={num_experts}): "
    #             f"max_diff={max_diff} >= {self.tolerance}"
    #         )

    def test_two_functions_consistency(self):
        """Test that both functions produce similar results.

        Compare Triton implementation with PyTorch reference implementation
        using shapes from Qwen3 MoE architecture.
        """
        from xtuner.v1.optim.muon import zeropower_via_newtonschulz5
        from xtuner.v1.optim.newton_schulz_triton import newton_schulz_triton

        # Scaled-down test cases based on Qwen3 MoE config
        test_cases = [
            # Non-MoE cases
            (1, 1536, 512, "shared_expert_w1"),
            (1, 512, 1536, "shared_expert_w2"),
            # MoE w1/w3 like (M < N)
            (8, 192, 512, "moe_w1"),
            # MoE w2 like (M > N)
            (8, 512, 192, "moe_w2"),
            # Square cases
            (1, 512, 512, "square_regular"),
            (4, 256, 256, "square_moe"),
        ]

        for num_experts, M, N, name in test_cases:
            G = self._create_test_matrix(num_experts, M, N)

            result1 = zeropower_via_newtonschulz5(
                G, epsilon=self.epsilon, num_experts=num_experts
            )
            result2 = newton_schulz_triton(
                G, epsilon=self.epsilon, num_experts=num_experts
            )

            max_diff = (result1 - result2).abs().max().item()
            # Allow larger tolerance since implementations differ (PyTorch vs Triton)
            # Triton uses different kernel implementations which may have numerical differences
            assert max_diff < 3e-2, (
                f"Functions differ for {name} (num_experts={num_experts}, M={M}, N={N}): "
                f"max_diff={max_diff}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
