"""CPU regressions for opt-in diagnostics (no training dependencies
required)."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


_spec = importlib.util.spec_from_file_location(
    "training_diagnostics", Path(__file__).parents[2] / "xtuner/v1/utils/diagnostics.py"
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
DiagnosticsConfig = _module.DiagnosticsConfig
TrainingDiagnostics = _module.TrainingDiagnostics


class TestTrainingDiagnostics:
    def test_disabled_has_no_side_effects(self, tmp_path):
        model = torch.nn.Linear(2, 2)
        original = model.forward
        diagnostic = TrainingDiagnostics(DiagnosticsConfig(), tmp_path / "disabled", 0)
        with patch.object(torch.cuda, "Event", side_effect=AssertionError("CUDA touched")):
            diagnostic.install(model)
            diagnostic.set_step(1)
            diagnostic.mark("forward_begin")
            diagnostic.close()
        assert model.forward == original
        assert diagnostic._thread is None
        assert not (tmp_path / "disabled").exists()

    def test_rank_and_inclusive_window(self, tmp_path):
        cfg = DiagnosticsConfig(enabled=True, start_step=3, end_step=4, ranks=[1])
        for rank in (0, 1):
            d = TrainingDiagnostics(cfg, tmp_path, rank)
            for step in range(1, 6):
                d.set_step(step)
                d.mark("forward_begin")
            d.close()
        records = [json.loads(line) for line in (tmp_path / "rank1.jsonl").read_text().splitlines()]
        assert [r["step"] for r in records] == [3, 4]
        assert not (tmp_path / "rank0.jsonl").exists()

    def test_compiled_bound_method_and_gradients(self, tmp_path):
        class Attention(torch.nn.Module):
            @torch.compile(backend="eager")
            def forward(self, x):
                return x.square()

        model = Attention()
        original = model.forward
        d = TrainingDiagnostics(DiagnosticsConfig(enabled=True, level="detailed"), tmp_path, 0)
        wrapper = d._wrap(original, "module.forward", gradients=True)
        assert "_torchdynamo_orig_callable" not in wrapper.__dict__
        d._patch(model, "forward", torch.compiler.disable(wrapper))
        # Both the inactive and active paths must preserve the bound self.
        for step in (0, 1):
            d.set_step(step)
            x = torch.tensor([2.0], requires_grad=True)
            model(x).sum().backward()
            assert torch.equal(x.grad, torch.tensor([4.0]))
        d.close()
        assert "forward" not in vars(model)
        assert model.forward == original

    def test_exception_and_staticmethod_restoration(self, tmp_path):
        class Operator:
            @staticmethod
            def forward(ctx, x):
                raise ValueError("test failure")

        original = vars(Operator)["forward"]
        d = TrainingDiagnostics(DiagnosticsConfig(enabled=True, level="detailed"), tmp_path, 0)
        d.set_step(1)
        d._patch(Operator, "forward", staticmethod(d._wrap(Operator.forward, "op.forward")))
        try:
            with pytest.raises(ValueError, match="test failure"):
                Operator.forward(SimpleNamespace(), 1)
        finally:
            d.close()
        assert vars(Operator)["forward"] is original
        records = [json.loads(line) for line in (tmp_path / "rank0.jsonl").read_text().splitlines()]
        assert any(r["kind"] == "cpu_error" for r in records)

    def test_invalid_configuration(self):
        with pytest.raises(ValueError):
            DiagnosticsConfig(start_step=4, end_step=3)
        with pytest.raises(ValueError):
            DiagnosticsConfig(ranks=[-1])
        with pytest.raises(ValueError):
            DiagnosticsConfig(cuda_events=True)

    def test_hook_cleanup_at_step_boundary(self, tmp_path):
        d = TrainingDiagnostics(DiagnosticsConfig(enabled=True, level="detailed"), tmp_path, 0)
        d.set_step(1)
        x = torch.tensor([2.0], requires_grad=True)
        wrapped = d._wrap(lambda value: value.square(), "square", gradients=True)
        y = wrapped(x)
        assert len(d._hooks) == 2
        y.sum().backward()
        assert torch.equal(x.grad, torch.tensor([4.0]))
        d.set_step(2)
        assert d._hooks == []
        assert not x._backward_hooks
        d.close()

    def test_cuda_event_capacity_without_real_cuda(self, tmp_path):
        cfg = DiagnosticsConfig(enabled=True, level="detailed", cuda_events=True, max_pending_cuda_events=1)
        d = TrainingDiagnostics(cfg, tmp_path, 0)
        event = SimpleNamespace(record=lambda stream: None)
        stream = SimpleNamespace(cuda_stream=123)
        with (
            patch.object(torch.cuda, "is_initialized", return_value=True),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "Event", return_value=event),
            patch.object(_module.threading.Thread, "start"),
            patch.object(_module.threading.Thread, "join"),
        ):
            d._event("first", {"call": 1}, stream)
            d._event("second", {"call": 2}, stream)
            assert len(d._pending) == 1
            assert d._pending[0][1]["call"] == 1
            d.close()
        records = [json.loads(line) for line in (tmp_path / "rank0.jsonl").read_text().splitlines()]
        assert sum(r["kind"] == "event_capacity_reached" for r in records) == 1
        assert records[-1]["kind"] == "pending_at_close"

    def test_logging_failure_preserves_original_exception(self, tmp_path):
        d = TrainingDiagnostics(DiagnosticsConfig(enabled=True, level="detailed"), tmp_path, 0)
        d.set_step(1)
        original_emit = d._emit

        def emit(kind, **fields):
            if kind == "cpu_error":
                raise OSError("disk full")
            original_emit(kind, **fields)

        with patch.object(d, "_emit", side_effect=emit):
            with pytest.raises(ValueError, match="original training error"):
                with d.span("forward"):
                    raise ValueError("original training error")
        d.close()
