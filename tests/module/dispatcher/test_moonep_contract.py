import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.model.moe.moe import MoEConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.dispatcher import NaiveDispatcher
from xtuner.v1.module.router import GreedyRouter, GreedyRouterConfig


def _moe_config(**overrides) -> MoEConfig:
    values = dict(
        vocab_size=128,
        max_position_embeddings=32,
        pad_token_id=0,
        eos_token_id=1,
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention=MHAConfig(
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=64,
        ),
        n_routed_experts=8,
        n_shared_experts=0,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        router=GreedyRouterConfig(
            scoring_func="softmax",
            norm_topk_prob=True,
            router_scaling_factor=1.0,
        ),
        compile_cfg=False,
    )
    values.update(overrides)
    return MoEConfig(**values)


def test_moonep_is_a_standard_model_config_choice() -> None:
    config = _moe_config(dispatcher="moonep", moonep_staging_reference=True)

    assert config.dispatcher == "moonep"
    assert config.moonep_staging_reference is True
    assert config.moonep_num_sms == 64
    assert config.intra_layer_micro_batch == 1


def test_moonep_rejects_fp8_at_model_construction() -> None:
    config = _moe_config(
        dispatcher="moonep",
        float8_cfg=Float8Config(scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE),
    )

    with pytest.raises(ValueError, match="requires BF16 expert compute; FP8 is not supported"):
        config.build()


def test_non_moonep_build_does_not_import_optional_backend() -> None:
    # A fresh interpreter with an explicit missing-module sentinel models an
    # XTuner installation that does not have MoonEP installed.
    source = """
import sys
sys.modules[\"moonep\"] = None
from xtuner.v1.module.dispatcher import NaiveDispatcher, build_dispatcher
dispatcher = build_dispatcher(None, n_routed_experts=4)
assert isinstance(dispatcher, NaiveDispatcher)
"""
    subprocess.run([sys.executable, "-c", source], check=True, capture_output=True, text=True)


def test_selecting_moonep_reports_the_missing_optional_backend() -> None:
    source = """
import sys
sys.modules["moonep"] = None
from types import SimpleNamespace
from xtuner.v1.module.dispatcher.moonep import MoonEPModelRuntime
try:
    MoonEPModelRuntime(
        ep_group=SimpleNamespace(size=lambda: 4),
        hidden_size=128,
        intermediate_size=128,
        num_experts=8,
        top_k=2,
        intra_layer_micro_batch=1,
        staging_reference=False,
    )
except RuntimeError as exc:
    assert "requires the MoonEP-mod integration package" in str(exc)
else:
    raise AssertionError("selecting MoonEP unexpectedly succeeded")
"""
    subprocess.run([sys.executable, "-c", source], check=True, capture_output=True, text=True)


def test_missing_grouped_gemm_does_not_disable_triton_backend() -> None:
    source = """
import sys
sys.modules["grouped_gemm"] = None
sys.modules["grouped_gemm_backend"] = None
from xtuner.v1.ops.moe.cuda import cutlass_group_gemm, triton_group_gemm
assert cutlass_group_gemm is None
assert callable(triton_group_gemm)
"""
    subprocess.run([sys.executable, "-c", source], check=True, capture_output=True, text=True)


def test_fully_shard_private_api_is_isolated_to_the_landing_module() -> None:
    dispatcher_dir = Path(__file__).parents[3] / "xtuner" / "v1" / "module" / "dispatcher"
    users = [
        path.name for path in dispatcher_dir.glob("*.py") if "torch.distributed.fsdp._fully_shard" in path.read_text()
    ]

    assert users == ["fsdp_vmm_landing.py"]


def test_existing_dispatcher_keeps_public_preprocess_behavior() -> None:
    dispatcher = NaiveDispatcher(n_routed_experts=4)
    hidden_states = torch.randn(3, 8)
    topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3]])
    topk_weights = torch.full((3, 2), 0.5)

    result = dispatcher.dispatch_preprocess(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        tokens_per_expert=torch.bincount(topk_ids.flatten(), minlength=4),
    )

    assert result["hidden_states"] is hidden_states
    assert result["topk_ids"] is topk_ids


def test_router_owns_logical_tokens_per_expert() -> None:
    router = GreedyRouter(
        n_routed_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
    )
    result = router(torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]]))

    assert set(result) == {"logits", "router_weights", "topk_weights", "topk_ids", "tokens_per_expert"}
    torch.testing.assert_close(
        result["tokens_per_expert"],
        torch.bincount(result["topk_ids"].flatten(), minlength=4),
    )


@pytest.mark.parametrize("width", [1, 2, 4])
def test_moe_list_forward_rejects_a_different_width(width: int) -> None:
    model = _moe_config(intra_layer_micro_batch=width).build()
    input_ids = torch.tensor([[2, 3, 4]])
    contexts = [
        SequenceContext.from_input_ids((input_ids.clone(),), device="cpu")
        for _ in range(width + 1)
    ]

    with pytest.raises(ValueError, match=f"width {width + 1} does not match configured width {width}"):
        model(
            seq_ctx=contexts,
            loss_ctx=[{} for _ in contexts],
        )


def test_runtime_meta_build_does_not_require_or_allocate_a_backend_workspace(monkeypatch) -> None:
    from xtuner.v1.module.dispatcher import moonep as moonep_integration
    from xtuner.v1.module.dispatcher.moonep import MoonEPModelRuntime
    from xtuner.v1.module.grouped_linear import moe_group_linear
    from xtuner.v1.ops.moe.cuda.group_gemm import triton_group_gemm

    # Workspace policy belongs to XTuner and allocation happens only after
    # FSDP installation, so the optional backend needs no workspace interface.
    backend = SimpleNamespace(
        __file__="/tmp/MoonEP-mod/moonep/__init__.py",
        XTUNER_INTEGRATION_API_VERSION=3,
        Buffer=object,
    )
    monkeypatch.setattr(moonep_integration, "_moonep_backend", backend)
    monkeypatch.setattr(moonep_integration, "_MOONEP_IMPORT_ERROR", None)
    monkeypatch.setattr(moe_group_linear, "group_gemm", triton_group_gemm)
    ep_group = SimpleNamespace(size=lambda: 4)

    runtime = MoonEPModelRuntime(
        ep_group=ep_group,
        hidden_size=128,
        intermediate_size=128,
        num_experts=8,
        top_k=2,
        intra_layer_micro_batch=2,
        staging_reference=False,
    )

    assert not hasattr(backend, "ExpertVMMWorkspace")
    assert isinstance(runtime, MoonEPModelRuntime)


def test_runtime_allows_triton_grouped_gemm(monkeypatch) -> None:
    from xtuner.v1.module.dispatcher import moonep as moonep_integration
    from xtuner.v1.module.dispatcher.moonep import MoonEPModelRuntime
    from xtuner.v1.module.grouped_linear import moe_group_linear
    from xtuner.v1.ops.moe.cuda.group_gemm import triton_group_gemm

    monkeypatch.setattr(
        moonep_integration,
        "_moonep_backend",
        SimpleNamespace(
            __file__="/tmp/MoonEP-mod/moonep/__init__.py", XTUNER_INTEGRATION_API_VERSION=3, Buffer=object
        ),
    )
    monkeypatch.setattr(moonep_integration, "_MOONEP_IMPORT_ERROR", None)
    monkeypatch.setattr(moe_group_linear, "group_gemm", triton_group_gemm)

    runtime = MoonEPModelRuntime(
        ep_group=SimpleNamespace(size=lambda: 4),
        hidden_size=128,
        intermediate_size=128,
        num_experts=8,
        top_k=2,
        intra_layer_micro_batch=1,
        staging_reference=False,
    )

    assert isinstance(runtime, MoonEPModelRuntime)


@pytest.mark.parametrize(
    ("environment_value", "effective_cutlass", "valid"),
    [(None, True, False), ("1", False, False), ("1", True, True)],
)
def test_runtime_requires_grouped_gemm_cutlass_backend(
    monkeypatch, environment_value: str | None, effective_cutlass: bool, valid: bool
) -> None:
    pytest.importorskip("grouped_gemm")
    from grouped_gemm import backend as grouped_gemm_backend

    from xtuner.v1.module.dispatcher import moonep as moonep_integration
    from xtuner.v1.module.dispatcher.moonep import MoonEPModelRuntime
    from xtuner.v1.module.grouped_linear import moe_group_linear
    from xtuner.v1.ops.moe.cuda import cutlass_group_gemm

    assert cutlass_group_gemm is not None
    monkeypatch.setattr(
        moonep_integration,
        "_moonep_backend",
        SimpleNamespace(
            __file__="/tmp/MoonEP-mod/moonep/__init__.py",
            XTUNER_INTEGRATION_API_VERSION=3,
            Buffer=object,
        ),
    )
    monkeypatch.setattr(moonep_integration, "_MOONEP_IMPORT_ERROR", None)
    monkeypatch.setattr(moe_group_linear, "group_gemm", cutlass_group_gemm)
    monkeypatch.setattr(grouped_gemm_backend, "use_cutlass", effective_cutlass)
    if environment_value is None:
        monkeypatch.delenv("GROUPED_GEMM_USE_CUTLASS", raising=False)
    else:
        monkeypatch.setenv("GROUPED_GEMM_USE_CUTLASS", environment_value)

    kwargs = dict(
        ep_group=SimpleNamespace(size=lambda: 4),
        hidden_size=128,
        intermediate_size=128,
        num_experts=8,
        top_k=2,
        intra_layer_micro_batch=1,
        staging_reference=False,
    )
    if valid:
        assert isinstance(MoonEPModelRuntime(**kwargs), MoonEPModelRuntime)
    else:
        with pytest.raises(RuntimeError, match="grouped_gemm requires GROUPED_GEMM_USE_CUTLASS=1"):
            MoonEPModelRuntime(**kwargs)


def test_runtime_reports_optional_backend_source_on_capability_mismatch(monkeypatch) -> None:
    from xtuner.v1.module.dispatcher import moonep as moonep_integration
    from xtuner.v1.module.dispatcher.moonep import MoonEPModelRuntime

    backend = SimpleNamespace(
        __file__="/wrong/worktree/moonep/__init__.py",
        XTUNER_INTEGRATION_API_VERSION=0,
    )
    monkeypatch.setattr(moonep_integration, "_moonep_backend", backend)
    monkeypatch.setattr(moonep_integration, "_MOONEP_IMPORT_ERROR", None)

    with pytest.raises(RuntimeError, match="/wrong/worktree/moonep/__init__.py"):
        MoonEPModelRuntime(
            ep_group=SimpleNamespace(size=lambda: 4),
            hidden_size=128,
            intermediate_size=128,
            num_experts=8,
            top_k=2,
            intra_layer_micro_batch=1,
            staging_reference=False,
        )
