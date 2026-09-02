import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
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
    from xtuner.v1.module.dispatcher.moonep import MoonEPRuntime

    # Workspace policy belongs to XTuner and allocation happens only after
    # FSDP installation, so the optional backend needs no workspace interface.
    backend = SimpleNamespace(
        __file__="/tmp/MoonEP-mod/moonep/__init__.py",
        XTUNER_INTEGRATION_API_VERSION=3,
        Buffer=object,
    )
    monkeypatch.setitem(sys.modules, "moonep", backend)
    ep_group = SimpleNamespace(size=lambda: 4)

    runtime = MoonEPRuntime(
        ep_group=ep_group,
        hidden_size=128,
        intermediate_size=128,
        num_experts=8,
        top_k=2,
        intra_layer_micro_batch=2,
        staging_reference=False,
    )

    assert not hasattr(backend, "ExpertVMMWorkspace")
    assert isinstance(runtime, MoonEPRuntime)


def test_runtime_reports_optional_backend_source_on_capability_mismatch(monkeypatch) -> None:
    from xtuner.v1.module.dispatcher.moonep import MoonEPRuntime

    backend = SimpleNamespace(
        __file__="/wrong/worktree/moonep/__init__.py",
        XTUNER_INTEGRATION_API_VERSION=0,
    )
    monkeypatch.setitem(sys.modules, "moonep", backend)

    with pytest.raises(RuntimeError, match="/wrong/worktree/moonep/__init__.py"):
        MoonEPRuntime(
            ep_group=SimpleNamespace(size=lambda: 4),
            hidden_size=128,
            intermediate_size=128,
            num_experts=8,
            top_k=2,
            intra_layer_micro_batch=1,
            staging_reference=False,
        )
