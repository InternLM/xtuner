from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from xtuner._testing.testcase import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.loss.moe_loss import BalancingLossConfig, ZLossConfig
from xtuner.v1.model.base import BaseModel, ModelItem, XTunerBaseModelConfig
from xtuner.v1.model.moe.moe import MoE, MoEConfig, SequenceContext
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router import NoAuxRouterConfig


class _ReduceSumToyConfig(XTunerBaseModelConfig):
    hidden_size: int = 8

    def build(self) -> "_ReduceSumToyModel":
        return _ReduceSumToyModel(self)


class _ReduceSumToyModel(BaseModel):
    config: _ReduceSumToyConfig

    def __init__(self, config: _ReduceSumToyConfig):
        super().__init__(config)
        self.fc = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self._init_load_spec()

    def to_hf_key_list(self, key: str) -> list[str]:
        return [key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestReduceSumGradient(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 2

    def test_bf16_reduce_sum_equals_local_grad_sum(self):
        # Regression guard for the torch 2.10 bf16 FSDP reduce-sum path: `set_gradient_reduce_sum`
        # must make the reduce-scatter yield the exact SUM of per-rank local gradients, not the
        # AVG default and not the all-zero result of the NCCL PreMulSum bf16 bug (the failure that
        # appears when only the divide factor is set). Assertions run in bf16 on purpose; fp32
        # reduction would mask the PreMulSum zeroing.
        self.create_pg("cuda")

        torch.manual_seed(0)
        dim = 8
        model = _ReduceSumToyConfig(hidden_size=dim, compile_cfg=False).build().cuda()
        # Gradient of `(x @ w.T).sum()` w.r.t. `w` is independent of the weight value, so the bf16
        # weight cast under FSDP does not perturb the reference; only the reduction dtype matters.
        ref_weight = model.fc.weight.detach().clone()

        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
        fully_shard(model, mp_policy=mp_policy)
        model.set_gradient_reduce_sum()

        rank = dist.get_rank()
        world = dist.get_world_size()
        # Distinct input per rank so local gradients differ; SUM and MEAN are then clearly separable.
        x = (torch.arange(dim, dtype=torch.float32, device="cuda") + rank + 1).reshape(1, dim)
        model(x).sum().backward()
        full_grad = model.fc.weight.grad.full_tensor().float()

        # Independent reference: this rank's local gradient on an unsharded copy, all-gathered.
        w = ref_weight.clone().detach().requires_grad_(True)
        (x @ w.T).sum().backward()
        g_local = w.grad.float()
        gathered = [torch.zeros_like(g_local) for _ in range(world)]
        dist.all_gather(gathered, g_local)
        g_sum = sum(gathered)
        g_mean = g_sum / world

        assert not torch.all(full_grad.abs() < 1e-9), "bf16 reduce-sum returned all-zero gradients"
        rel_to_sum = ((full_grad - g_sum).norm() / (g_sum.norm() + 1e-12)).item()
        rel_to_mean = ((full_grad - g_mean).norm() / (g_mean.norm() + 1e-12)).item()
        assert rel_to_sum < 1e-2, f"expected SUM of local grads, rel_to_sum={rel_to_sum}"
        assert rel_to_mean > 0.1, f"gradient matched MEAN, reduce-sum not applied, rel_to_mean={rel_to_mean}"


@contextmanager
def _fake_dist_uninitialized():
    # Make the single-process reference skip its WORLD all_reduce so the CE denominator counts each
    # token once, i.e. a plain token-mean CE over the whole (concatenated) global batch.
    orig = dist.is_initialized
    dist.is_initialized = lambda: False
    try:
        yield
    finally:
        dist.is_initialized = orig


def _tiny_moe_config(ep_size: int, balancing: bool, z_loss: bool) -> MoEConfig:
    return MoEConfig(
        vocab_size=1024,
        max_position_embeddings=512,
        pad_token_id=0,
        eos_token_id=0,
        num_hidden_layers=2,
        hidden_size=256,
        intermediate_size=512,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        hidden_act="silu",
        attention=MHAConfig(num_attention_heads=8, num_key_value_heads=8, head_dim=32),
        tie_word_embeddings=False,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        hidden_factor=1.0,
        moe_intermediate_size=256,
        compile_cfg=False,
        router=NoAuxRouterConfig(
            scoring_func="sigmoid", router_scaling_factor=1.0, n_group=1, topk_group=1, norm_topk_prob=True
        ),
        ep_size=ep_size,
        balancing_loss_cfg=BalancingLossConfig() if balancing else None,
        z_loss_cfg=ZLossConfig() if z_loss else None,
    )


def _full_grads(model) -> dict[str, torch.Tensor]:
    out = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        g = g.full_tensor() if isinstance(g, DTensor) else g
        out[name.replace("._checkpoint_wrapped_module", "")] = g.detach().float().cpu()
    return out


class TestReduceSumEndToEnd(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 2

    def test_reduce_sum_moe_matches_token_mean_reference_ep1(self):
        # EP=1 keeps every param replicated, so scale_and_reduce_grad's no-divide SUM branch is on
        # the critical path.
        self._check_token_mean_parity(ep_size=1)

    def test_reduce_sum_moe_matches_token_mean_reference_ep2(self):
        # EP=2 exercises the expert path: removing the expert div_(ep_size) (the §6 high-risk change)
        # must still reproduce the token-mean reference. The experts_fsdp reduce-scatter is the only
        # aggregation their grads get, so a wrong factor here would show as an ep_size scaling.
        self._check_token_mean_parity(ep_size=2)

    def _check_token_mean_parity(self, ep_size: int):
        # The reduce-sum path (FSDP SUM reduce-scatter + no CE WORLD all_reduce + SUM-only
        # scale_and_reduce_grad) must reproduce a single-process, full-batch token-mean CE gradient.
        # grad-acc=2 exercises SUM composability across micro-batch backwards.
        self.create_pg("cuda")
        device = "cuda"
        rank = dist.get_rank()
        seq_len = 32
        n_microbatch = 2
        efsdp = self.world_size // ep_size

        config = _tiny_moe_config(ep_size=ep_size, balancing=False, z_loss=False)
        torch.manual_seed(0)
        fsdp_cfg = FSDPConfig(cpu_offload=False, ep_size=ep_size, reduce_dtype=torch.bfloat16)
        engine = TrainEngine(model_cfg=config, optim_cfg=AdamWConfig(), fsdp_cfg=fsdp_cfg)
        engine.init_model_weights()

        gold_weights = {
            name.replace("._checkpoint_wrapped_module", ""): (
                p.full_tensor() if isinstance(p, DTensor) else p.detach()
            )
            .detach()
            .float()
            .cpu()
            for name, p in engine.model.named_parameters()
        }

        # One distinct sequence per (fsdp shard, micro-batch); concatenated they form the reference
        # global batch. EP replicas (same fsdp position) consume identical data.
        gen = torch.Generator().manual_seed(1234)
        shards = [torch.randint(0, 512, (1, seq_len + 1), generator=gen) for _ in range(efsdp * n_microbatch)]

        def build_items(ids_list):
            loss_data = []
            for ids in ids_list:
                ids = ids.to(device)
                seq_ctx = SequenceContext.from_input_ids(input_ids=(ids[:, :-1],), device=device)
                loss_data.append({"seq_ctx": seq_ctx, "shifted_labels": ids[:, 1:]})
            loss_ctx_list = engine.model.build_loss_ctx_batch(loss_data, sp_mesh=None)
            return [ModelItem(seq_ctx=d["seq_ctx"], loss_ctx=lc) for d, lc in zip(loss_data, loss_ctx_list)]

        fsdp_idx = rank // ep_size
        my_shards = shards[fsdp_idx * n_microbatch : (fsdp_idx + 1) * n_microbatch]
        engine.model.zero_grad(set_to_none=True)
        engine.train_step(build_items(my_shards))
        engine.model.scale_and_reduce_grad()
        dist_grads = _full_grads(engine.model)

        if rank == 0:
            ref = MoE(config=_tiny_moe_config(ep_size=1, balancing=False, z_loss=False)).to(torch.bfloat16).to(device)
            missing, unexpected = ref.load_state_dict(
                {k: v.to(torch.bfloat16).to(device) for k, v in gold_weights.items()}, strict=False
            )
            assert not unexpected, f"unexpected keys: {unexpected}"
            ref.zero_grad(set_to_none=True)
            loss_cfg = CELossConfig()
            seq_ctxs, loss_ctxs = [], []
            for ids in shards:
                ids = ids.to(device)
                seq_ctxs.append(SequenceContext.from_input_ids(input_ids=(ids[:, :-1],), device=device))
                loss_ctxs.append(loss_cfg.build(data={"shifted_labels": ids[:, 1:]}, sp_mesh=None))
            with _fake_dist_uninitialized():
                loss_ctxs = loss_cfg.loss_ctx_cls.build_batches(loss_ctxs)
                total = torch.zeros((), device=device)
                for seq_ctx, lc in zip(seq_ctxs, loss_ctxs):
                    total = total + ref(seq_ctx=seq_ctx, loss_ctx={"lm": lc})["loss"]
                total.backward()
            ref_grads = {n: p.grad.detach().float().cpu() for n, p in ref.named_parameters() if p.grad is not None}

            ratios = []
            for name, rg in ref_grads.items():
                dg = dist_grads.get(name)
                if dg is None or dg.shape != rg.shape:
                    continue
                ratios.append((dg.norm() / rg.norm().clamp_min(1e-12)).item())
            ratios_t = torch.tensor(ratios)
            median = ratios_t.median().item()
            assert abs(median - 1.0) < 0.05, f"ep={ep_size} reduce-sum grad norm ratio median={median} (want ~1.0)"
            assert (ratios_t - 1.0).abs().max().item() < 0.3, f"ep={ep_size} a param grad ratio drifted: {ratios_t}"

    def test_reduce_sum_with_aux_losses_produces_finite_grads(self):
        # Balancing + z-loss enabled: after dropping their `× world_size` injection, backward must
        # still produce finite, non-zero gradients (the aux-loss backward flows through the router).
        self.create_pg("cuda")
        device = "cuda"
        rank = dist.get_rank()
        seq_len = 32

        config = _tiny_moe_config(ep_size=1, balancing=True, z_loss=True)
        torch.manual_seed(0)
        fsdp_cfg = FSDPConfig(cpu_offload=False, ep_size=1, reduce_dtype=torch.bfloat16)
        engine = TrainEngine(model_cfg=config, optim_cfg=AdamWConfig(), fsdp_cfg=fsdp_cfg)
        engine.init_model_weights()

        gen = torch.Generator().manual_seed(1234 + rank)
        loss_data = []
        for _ in range(2):
            ids = torch.randint(0, 512, (1, seq_len + 1), generator=gen).to(device)
            seq_ctx = SequenceContext.from_input_ids(input_ids=(ids[:, :-1],), device=device)
            loss_data.append({"seq_ctx": seq_ctx, "shifted_labels": ids[:, 1:]})
        loss_ctx_list = engine.model.build_loss_ctx_batch(loss_data, sp_mesh=None)
        items = [ModelItem(seq_ctx=d["seq_ctx"], loss_ctx=lc) for d, lc in zip(loss_data, loss_ctx_list)]

        engine.model.zero_grad(set_to_none=True)
        info = engine.train_step(items)
        engine.model.scale_and_reduce_grad()

        assert torch.isfinite(torch.tensor(info["total_loss"])), "total_loss is not finite"
        grads = _full_grads(engine.model)
        gate_grads = [g for n, g in grads.items() if "gate" in n]
        assert gate_grads, "router gate has no gradient; aux-loss backward path is broken"
        for name, g in grads.items():
            assert torch.isfinite(g).all(), f"non-finite gradient in {name}"
        assert any(g.abs().sum() > 0 for g in gate_grads), "router gate gradients are all zero"


class TestBalancingLossReduceSum(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        # >1 rank forms the replicate group over which the router/gate gradient is aggregated.
        return 4

    def test_balancing_local_sum_matches_global(self):
        # Isolate the balancing-loss gate gradient (CE off) and prove the reduce-sum rewrite is
        # exact for it: aggregate grad-norm parity cannot single out this term. The new scheme
        # (local_gating_sum + global detached coefficients, gradients SUM-reduced across ranks)
        # must reproduce the old scheme (all_reduce_autograd + FSDP mean-reduce) element-wise. In
        # fp32 with a fixed router this is a clean equality (no bf16 routing jitter), so the
        # tolerance is tight; if it fails, the "linear + global detached coefficient => local-sum
        # equals global" reasoning is wrong and the reduce-sum switch is unsound.
        from torch.distributed.nn.functional import all_reduce as all_reduce_autograd

        self.create_pg("cuda")
        device = "cuda"
        rank = dist.get_rank()
        world = dist.get_world_size()

        n_layers, n_experts, topk, n_tokens, hidden = 2, 8, 2, 16, 32
        # Replicated gate: identical init on every rank. Distinct per-rank token features so each
        # rank owns a different local_gating_sum, exactly the asymmetry the reduce path must handle.
        w_gen = torch.Generator().manual_seed(123)
        w0 = torch.randn(hidden, n_experts, generator=w_gen, dtype=torch.float32).to(device)
        x_gen = torch.Generator().manual_seed(1000 + rank)
        x = torch.randn(n_layers, n_tokens, hidden, generator=x_gen, dtype=torch.float32).to(device)

        def router_weights(w: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(torch.einsum("lth,he->lte", x, w))

        # Non-differentiable per-expert token counts; global view via a detached SUM all_reduce.
        rw_detached = router_weights(w0)
        _, selected = torch.topk(rw_detached, topk, dim=-1)
        tpe_local = torch.stack(
            [torch.histc(selected[layer].float(), bins=n_experts, min=0, max=n_experts) for layer in range(n_layers)]
        ).long()
        tpe_global = tpe_local.clone()
        dist.all_reduce(tpe_global, op=dist.ReduceOp.SUM)

        cfg = BalancingLossConfig()
        alpha = cfg.balancing_loss_alpha
        tokens_global = tpe_global.sum(-1)
        seqlen_global = tokens_global // topk
        scale_global = n_experts / tokens_global

        # New scheme via the real BalancingLossContext: per-rank local loss, gradients SUM-reduced.
        w_new = w0.clone().requires_grad_(True)
        rw_new = router_weights(w_new)
        ctx = cfg.build()
        for layer in range(n_layers):
            ctx.accumulate(router_weights=rw_new[layer])
        loss_new, _ = ctx.finalize(
            tokens_per_expert_local=tpe_local,
            tokens_per_expert_global=tpe_global,
            n_routed_experts=n_experts,
            num_experts_per_tok=topk,
            non_pad_token=n_tokens,
        )
        loss_new.backward()
        g_new = w_new.grad.clone()
        dist.all_reduce(g_new, op=dist.ReduceOp.SUM)  # FSDP/scale_and_reduce SUM over the replicate group

        # Old scheme: all_reduce_autograd over the gating sum, then FSDP mean-reduce (divide by world).
        w_old = w0.clone().requires_grad_(True)
        rw_old = router_weights(w_old)
        gating_sum = torch.stack([rw_old[layer].sum(dim=0) for layer in range(n_layers)])
        routing_weights_sum_global = all_reduce_autograd(gating_sum, op=dist.ReduceOp.SUM)
        routing_weights_mean = routing_weights_sum_global / seqlen_global.unsqueeze(-1)
        loss_old = (scale_global * (tpe_global * routing_weights_mean).sum(-1)).sum() * alpha
        loss_old.backward()
        g_old = w_old.grad.clone()
        g_old.div_(world)  # FSDP mean-reduce (old default AVG)
        dist.all_reduce(g_old, op=dist.ReduceOp.SUM)

        if rank == 0:
            assert g_new.abs().sum() > 0, "balancing gate gradient is all zero"
            torch.testing.assert_close(g_new, g_old, rtol=1e-4, atol=1e-4)

    def test_zloss_local_matches_global(self):
        # Same isolation for z-loss (CE off). The new scheme drops the `× world_size` factor and
        # relies on the SUM gradient reduction across the replicate group; it must reproduce the old
        # scheme (`× world_size` in the loss + FSDP mean-reduce) on the router gate, element-wise.
        self.create_pg("cuda")
        device = "cuda"
        rank = dist.get_rank()
        world = dist.get_world_size()

        n_layers, n_experts, n_tokens, hidden = 2, 8, 16, 32
        w_gen = torch.Generator().manual_seed(321)
        w0 = torch.randn(hidden, n_experts, generator=w_gen, dtype=torch.float32).to(device)
        x_gen = torch.Generator().manual_seed(2000 + rank)
        x = torch.randn(n_layers, n_tokens, hidden, generator=x_gen, dtype=torch.float32).to(device)

        def router_logits(w: torch.Tensor) -> torch.Tensor:
            return torch.einsum("lth,he->lte", x, w)

        num_tokens_local = n_tokens
        num_tokens_global = torch.tensor(num_tokens_local, device=device, dtype=torch.int64)
        dist.all_reduce(num_tokens_global, op=dist.ReduceOp.SUM)  # detached global token count
        denom_global = torch.clamp(num_tokens_global, min=1)

        cfg = ZLossConfig()
        alpha = cfg.z_loss_alpha

        # New scheme via the real ZLossContext: per-layer local z-loss (no × world_size), SUM-reduced.
        w_new = w0.clone().requires_grad_(True)
        logits_new = router_logits(w_new)
        ctx = cfg.build()
        z_new = torch.zeros((), device=device)
        for layer in range(n_layers):
            z_new = z_new + ctx.accumulate(
                router_logits=logits_new[layer],
                num_tokens_local=num_tokens_local,
                num_tokens_global=num_tokens_global,
            )
        z_new.backward()
        g_new = w_new.grad.clone()
        dist.all_reduce(g_new, op=dist.ReduceOp.SUM)

        # Old scheme: the same per-layer z-loss but multiplied by world_size, then FSDP mean-reduce.
        w_old = w0.clone().requires_grad_(True)
        logits_old = router_logits(w_old)
        z_old = torch.zeros((), device=device)
        for layer in range(n_layers):
            base = torch.logsumexp(logits_old[layer], dim=-1).square().sum() / max(num_tokens_local, 1)
            z_old = z_old + base * num_tokens_local * world / denom_global * alpha
        z_old.backward()
        g_old = w_old.grad.clone()
        g_old.div_(world)  # FSDP mean-reduce (old default AVG)
        dist.all_reduce(g_old, op=dist.ReduceOp.SUM)

        if rank == 0:
            assert g_new.abs().sum() > 0, "z-loss gate gradient is all zero"
            torch.testing.assert_close(g_new, g_old, rtol=1e-4, atol=1e-4)


def _fsdp_reduce_cfg(module: FSDPModule):
    # Read back the reduce-scatter reduction config that set_gradient_divide_factor /
    # set_force_sum_reduction_for_comms write onto the FSDP param group.
    param_group = module._get_fsdp_state()._fsdp_param_group  # type: ignore[attr-defined]
    return param_group.gradient_divide_factor, param_group.force_sum_reduction_for_comms


class TestComposeReduceSumHook(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 2

    def test_root_pass_sets_sum_on_independently_sharded_children(self):
        # Compose/VLM models shard vision_tower / multi_modal_projector via their own fully_shard
        # overrides that do NOT set reduce-sum, plus a root self._fully_shard wrap. The fix relies on
        # a single root-level set_gradient_reduce_sum() covering every nested FSDPModule via
        # self.modules(). This reproduces that topology with independently sharded children and
        # asserts they all flip from the FSDP AVG default to divide_factor=1 + force_sum.
        self.create_pg("cuda")
        model = _ReduceSumToyConfig(hidden_size=8, compile_cfg=False).build().cuda()
        # Stand-ins for vision_tower / multi_modal_projector, sharded independently without reduce-sum.
        vision_like = nn.Linear(8, 8, bias=False).cuda()
        projector_like = nn.Linear(8, 8, bias=False).cuda()
        model.add_module("vision_like", vision_like)
        model.add_module("projector_like", projector_like)

        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
        fully_shard(vision_like, mp_policy=mp_policy)
        fully_shard(projector_like, mp_policy=mp_policy)
        fully_shard(model, mp_policy=mp_policy)  # root wrap, FSDP AVG default

        fsdp_modules = [m for m in model.modules() if isinstance(m, FSDPModule)]
        assert len(fsdp_modules) >= 3, "expected root + two independently sharded children"
        # Precondition: none are reduce-sum yet (default AVG has divide_factor None / force_sum False).
        for m in fsdp_modules:
            factor, force_sum = _fsdp_reduce_cfg(m)
            assert not (factor == 1.0 and force_sum), "precondition failed: module already reduce-sum"

        # The compose fix: one root-level pass.
        model.set_gradient_reduce_sum()

        for m in fsdp_modules:
            factor, force_sum = _fsdp_reduce_cfg(m)
            assert factor == 1.0 and force_sum is True, (
                f"nested FSDPModule not switched to SUM: divide_factor={factor} force_sum={force_sum}"
            )
