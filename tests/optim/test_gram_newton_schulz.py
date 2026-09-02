# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any

import pytest
import torch

from xtuner.v1.config.optim import MuonConfig
from xtuner.v1.optim import Muon
from xtuner.v1.optim import gram_newton_schulz as gram_ns_module
from xtuner.v1.optim.gram_newton_schulz import (
    POLAR_EXPRESS_COEFFICIENTS,
    POLAR_EXPRESS_SAFETY_FACTOR,
    GramNewtonSchulz,
)


def _cpu_orthogonalizer(**kwargs: Any) -> GramNewtonSchulz:
    return GramNewtonSchulz(use_kernels=False, compile_kwargs=None, **kwargs)


class TestGramNewtonSchulz:
    def test_polar_express_coefficients_include_recommended_safety_factor(self) -> None:
        unmodified_first = (8.28721201814563, -23.595886519098837, 17.300387312530933)
        expected = (
            unmodified_first[0] / POLAR_EXPRESS_SAFETY_FACTOR,
            unmodified_first[1] / POLAR_EXPRESS_SAFETY_FACTOR**3,
            unmodified_first[2] / POLAR_EXPRESS_SAFETY_FACTOR**5,
        )

        assert len(POLAR_EXPRESS_COEFFICIENTS) == 5
        torch.testing.assert_close(torch.tensor(POLAR_EXPRESS_COEFFICIENTS[0]), torch.tensor(expected))

    def test_adaptive_selection_falls_back_for_square_and_low_aspect_ratio(self) -> None:
        orthogonalizer = _cpu_orthogonalizer(min_aspect_ratio=2.0)

        assert orthogonalizer._selection(256, 256) == ("standard_fallback", "torch")
        assert orthogonalizer._selection(256, 512) == ("standard_fallback", "torch")
        assert orthogonalizer._selection(256, 768) == ("gram", "torch")
        assert orthogonalizer._selection(768, 256) == ("gram", "torch")

        orthogonalizer._kernel_backend = object()
        assert orthogonalizer._selection(256, 768) == ("gram", "cutedsl")

    def test_gram_matches_recommended_standard_path_on_cpu(self) -> None:
        torch.manual_seed(0)
        inputs = (torch.randn(12, 4), torch.randn(4, 12))
        gram = _cpu_orthogonalizer(min_aspect_ratio=2.0)
        standard = _cpu_orthogonalizer(min_aspect_ratio=100.0)

        for input_ in inputs:
            gram_output = gram(input_)
            standard_output = standard(input_)
            assert gram_output.dtype == input_.dtype
            assert torch.isfinite(gram_output).all()
            torch.testing.assert_close(gram_output, standard_output, atol=1e-2, rtol=1e-2)

    def test_concatenated_experts_are_orthogonalized_independently(self) -> None:
        torch.manual_seed(1)
        orthogonalizer = _cpu_orthogonalizer(min_aspect_ratio=2.0)
        experts = torch.randn(3, 12, 4, dtype=torch.float32)
        concatenated = experts.view(-1, experts.size(-1))

        batched = orthogonalizer(concatenated, num_experts=experts.size(0))
        independent = torch.cat([orthogonalizer(expert) for expert in experts])

        assert batched.shape == concatenated.shape
        assert batched.dtype == concatenated.dtype
        torch.testing.assert_close(batched, independent)

    def test_rejects_invalid_expert_layout(self) -> None:
        orthogonalizer = _cpu_orthogonalizer()

        with pytest.raises(ValueError, match="Invalid num_experts"):
            orthogonalizer(torch.randn(12, 4), num_experts=5)


class TestMuonGramNewtonSchulz:
    def test_config_keeps_gram_ns_opt_in(self) -> None:
        config = MuonConfig()

        assert config.use_gram_newton_schulz is False
        assert config.gram_restart_iterations == (2,)
        assert config.gram_ns_epsilon == 1e-7
        assert config.gram_ns_min_aspect_ratio == 2.0
        assert config.gram_ns_torch_compile is True

    def test_switch_selects_gram_ns_callback_without_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gram_ns_module, "_make_kernel_backend", lambda: None)
        parameter = torch.nn.Parameter(torch.randn(12, 4))

        optimizer = Muon(
            [parameter],
            use_gram_newton_schulz=True,
            gram_restart_iterations=(2,),
            gram_ns_min_aspect_ratio=2.0,
            gram_ns_torch_compile=False,
        )

        assert isinstance(optimizer._newton_schulz_func, GramNewtonSchulz)
        assert optimizer._newton_schulz_func.restart_iterations == (2,)

    def test_switch_rejects_other_newton_schulz_implementations(self) -> None:
        parameter = torch.nn.Parameter(torch.randn(12, 4))

        with pytest.raises(ValueError, match="cannot be combined"):
            Muon([parameter], use_gram_newton_schulz=True, use_triton=True)
