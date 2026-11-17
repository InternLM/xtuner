import pytest
import torch

from xtuner.v1.module.router.greedy import GreedyGroupedRouter, GreedyRouter
from xtuner.v1.module.router.noaux_router import NoAuxGroupedRouter, NoAuxRouter


ROUTER_KINDS = ("greedy", "greedy_grouped", "noaux", "noaux_grouped")


def _build_router(kind: str, device: torch.device):
    if kind == "greedy":
        router = GreedyRouter(
            n_routed_experts=16,
            num_experts_per_tok=2,
            norm_topk_prob=True,
        )
    elif kind == "greedy_grouped":
        router = GreedyGroupedRouter(
            n_routed_experts=16,
            num_experts_per_tok=8,
            router_n_groups=4,
            norm_topk_prob=True,
        )
    elif kind == "noaux":
        router = NoAuxRouter(
            n_routed_experts=16,
            num_experts_per_tok=2,
            router_scaling_factor=1.0,
            scoring_func="sigmoid",
            n_group=1,
            topk_group=1,
        )
    elif kind == "noaux_grouped":
        router = NoAuxGroupedRouter(
            n_routed_experts=16,
            num_experts_per_tok=8,
            router_scaling_factor=1.0,
            router_n_groups=4,
            scoring_func="sigmoid",
            n_group=1,
            topk_group=1,
        )
    else:
        raise AssertionError(f"unknown router kind: {kind}")

    router = router.to(device)
    if isinstance(router, NoAuxRouter):
        router.e_score_correction_bias.zero_()
    return router


@pytest.mark.parametrize("kind", ROUTER_KINDS)
def test_router_returns_fixed_device_counts(kind: str) -> None:
    device = torch.device("cpu")
    router = _build_router(kind, device)
    logits = torch.arange(48, dtype=torch.float32, device=device).view(3, 16)
    top_k = router.top_k
    routed_experts = torch.arange(3 * top_k, device=device).view(3, top_k).remainder(16)

    result = router(logits, rollout_routed_experts=routed_experts)
    expected = torch.bincount(routed_experts.flatten(), minlength=16)

    assert result["tokens_per_expert"].shape == (16,)
    assert result["tokens_per_expert"].dtype == torch.int64
    assert result["tokens_per_expert"].device == logits.device
    assert torch.equal(result["tokens_per_expert"], expected)
    assert result["tokens_per_expert"].sum() == routed_experts.numel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA profiler regression requires a GPU")
@pytest.mark.parametrize("kind", ROUTER_KINDS)
def test_router_counting_has_no_host_synchronization(kind: str) -> None:
    device = torch.device("cuda")
    router = _build_router(kind, device)
    logits = torch.randn(512, 16, dtype=torch.float32, device=device)
    routed_experts = torch.arange(
        512 * router.top_k,
        dtype=torch.int64,
        device=device,
    ).view(512, router.top_k).remainder(16)

    # Warm allocations before profiling so the marker contains only the real
    # router path, including its device-side count production.
    router(logits, rollout_routed_experts=routed_experts)
    torch.cuda.synchronize()
    marker = f"{kind}_router_forward"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile, torch.profiler.record_function(marker):
        result = router(logits, rollout_routed_experts=routed_experts)
    torch.cuda.synchronize()

    blocking_events = []
    for event in profile.events():
        parent = event.cpu_parent
        while parent is not None and parent.name != marker:
            parent = parent.cpu_parent
        if parent is not None and (
            "synchronize" in event.name.lower()
            or event.name in {"aten::item", "aten::_local_scalar_dense"}
        ):
            blocking_events.append(event.name)

    assert not blocking_events
    assert result["tokens_per_expert"].sum() == routed_experts.numel()
