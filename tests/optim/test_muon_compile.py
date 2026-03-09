# Copyright (c) OpenMMLab. All rights reserved.
"""Test Muon optimizer Newton-Schulz functions with/without torch.compile.

This test verifies that the Newton-Schulz orthogonalization produces consistent
results whether torch.compile is enabled or disabled.
"""

import pytest
import torch

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

    def test_zeropower_via_newtonschulz5_compile(self):
        """Test muon.zeropower_via_newtonschulz5 with/without compile."""
        from xtuner.v1.optim.muon import zeropower_via_newtonschulz5

        # Test both num_experts=1 and num_experts>1
        test_cases = [
            (1, 64, 32, "regular_matrix"),
            (4, 64, 32, "moe_matrix"),
            (1, 64, 64, "square_regular"),
            (8, 32, 32, "square_moe"),
        ]

        for num_experts, M, N, name in test_cases:
            G = self._create_test_matrix(num_experts, M, N)

            # Without compile
            result_no_compile = zeropower_via_newtonschulz5(
                G, epsilon=self.epsilon, num_experts=num_experts
            )

            # With compile
            compiled_fn = torch.compile(zeropower_via_newtonschulz5, fullgraph=True)
            result_compile = compiled_fn(G, epsilon=self.epsilon, num_experts=num_experts)

            # Compare results
            max_diff = (result_no_compile - result_compile).abs().max().item()
            assert max_diff < self.tolerance, (
                f"{name} (num_experts={num_experts}): max_diff={max_diff} >= {self.tolerance}"
            )

    def test_newton_schulz_triton(self):
        """Test newton_schulz_triton (without compile - Triton kernels don't support compile)."""
        from xtuner.v1.optim.newton_schulz_triton import newton_schulz_triton

        # Test both num_experts=1 and num_experts>1
        test_cases = [
            (1, 64, 32, "regular_matrix"),
            (4, 64, 32, "moe_matrix"),
            (1, 64, 64, "square_regular"),
            (8, 32, 32, "square_moe"),
        ]

        for num_experts, M, N, name in test_cases:
            G = self._create_test_matrix(num_experts, M, N)

            # Just verify it runs without error (Triton doesn't support torch.compile)
            result = newton_schulz_triton(G, epsilon=self.epsilon, num_experts=num_experts)
            
            # Basic sanity check: output should have correct shape
            assert result.shape == G.shape, f"{name}: output shape mismatch"
            
            # Output should not be all zeros or contain NaN/Inf
            assert not torch.isnan(result).any(), f"{name}: output contains NaN"
            assert not torch.isinf(result).any(), f"{name}: output contains Inf"
            assert result.abs().max() > 0, f"{name}: output is all zeros"

    def test_transpose_case_compile(self):
        """Test matrices where rows > cols (transpose case) with compile."""
        from xtuner.v1.optim.muon import zeropower_via_newtonschulz5

        test_cases = [
            (1, 100, 30, "transpose_regular"),
            (4, 100, 30, "transpose_moe"),
        ]

        for num_experts, M, N, name in test_cases:
            G = self._create_test_matrix(num_experts, M, N)

            # Without compile
            result_no_compile = zeropower_via_newtonschulz5(
                G, epsilon=self.epsilon, num_experts=num_experts
            )

            # With compile
            compiled_fn = torch.compile(zeropower_via_newtonschulz5, fullgraph=True)
            result_compile = compiled_fn(G, epsilon=self.epsilon, num_experts=num_experts)

            # Compare results
            max_diff = (result_no_compile - result_compile).abs().max().item()
            assert max_diff < self.tolerance, (
                f"zeropower_via_newtonschulz5 {name} (num_experts={num_experts}): "
                f"max_diff={max_diff} >= {self.tolerance}"
            )

    def test_two_functions_consistency(self):
        """Test that both functions produce similar results."""
        from xtuner.v1.optim.muon import zeropower_via_newtonschulz5
        from xtuner.v1.optim.newton_schulz_triton import newton_schulz_triton

        test_cases = [
            (1, 64, 32),
            (4, 64, 32),
            (1, 64, 64),
            (8, 32, 32),
        ]

        for num_experts, M, N in test_cases:
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
            assert max_diff < 2e-2, (
                f"Functions differ for (num_experts={num_experts}, M={M}, N={N}): "
                f"max_diff={max_diff}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
