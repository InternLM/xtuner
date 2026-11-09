import gc
import weakref
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from xtuner.v1.model import base
from xtuner.v1.model.base import BaseModel, XTunerBaseModelConfig
from xtuner.v1.utils import HFCheckpointLoader


class _TinyModel(BaseModel):
    def __init__(self) -> None:
        super().__init__(XTunerBaseModelConfig(compile_cfg=False))
        self.weight = nn.Parameter(torch.zeros(2))
        self.register_buffer("scale", torch.zeros(2))
        self.register_buffer("ignored", torch.zeros(2), persistent=False)
        self.child = nn.Linear(2, 2, bias=False)
        self.child.nested = nn.Linear(2, 2, bias=False)
        self.sibling = nn.Linear(2, 2, bias=False)
        self._init_load_spec()

    def to_hf_key_list(self, key: str) -> list[str]:
        return [key]

    _model_hf_keys = to_hf_key_list


class TestHFLoaderLifetime:
    @pytest.mark.parametrize("outcome", ["success", "load_error", "missing_tensor"])
    def test_releases_loader_without_cyclic_gc(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        outcome: Literal["success", "load_error", "missing_tensor"],
    ) -> None:
        model = _TinyModel()
        model_ref = weakref.ref(model)
        weights = {name: torch.full_like(param, 3) for name, param in model.state_dict().items()}
        save_file(weights, tmp_path / "model.safetensors")
        loader_refs: list[weakref.ReferenceType[HFCheckpointLoader]] = []
        events: list[str] = []

        class TrackingLoader(HFCheckpointLoader):
            def __init__(self, model_path: str) -> None:
                super().__init__(model_path)
                loader_refs.append(weakref.ref(self))

            def load(self, key: str) -> torch.Tensor | None:
                assert not torch.is_grad_enabled()
                events.append(key)
                if key == "child.weight":
                    if outcome == "load_error":
                        raise RuntimeError("injected load failure")
                    if outcome == "missing_tensor":
                        return None
                return super().load(key)

        # Exercise real checkpoint I/O and traversal on CPU; only GPU profiling
        # and synchronization need substitutes.
        monkeypatch.setattr(base, "HFCheckpointLoader", TrackingLoader)
        monkeypatch.setattr(base, "profile_time_and_memory", lambda _: nullcontext())
        monkeypatch.setattr(torch.accelerator, "synchronize", lambda: events.append("accelerator_sync"))
        monkeypatch.setattr(torch.cpu, "synchronize", lambda: events.append("cpu_sync"))

        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            if outcome == "success":
                assert model.from_hf(tmp_path) == (set(weights), set(), set())
                for name, param in model.state_dict().items():
                    torch.testing.assert_close(param, weights[name])
            else:
                message = "injected load failure" if outcome == "load_error" else "Internal error, Missing keys"
                # Do not retain ExceptionInfo / its traceback across the weakref checks.
                with pytest.raises(RuntimeError, match=message):
                    model.from_hf(tmp_path)

            expected_order = ["weight", "scale", "child.weight", "child.nested.weight", "sibling.weight"]
            if outcome == "load_error":
                assert events == expected_order[:3]
            else:
                assert events == [*expected_order, "accelerator_sync", "cpu_sync"]
            assert len(loader_refs) == 1
            # Keep the model alive while checking that from_hf released its loader.
            assert loader_refs[0]() is None
            del model
            # PrecisionPolicy holds a reference back to its model. Unlike the
            # loader above, model destruction therefore requires cyclic GC.
            gc.collect()
            assert model_ref() is None
        finally:
            if gc_was_enabled:
                gc.enable()
            gc.collect()
