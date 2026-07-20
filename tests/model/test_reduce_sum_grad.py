from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard

from xtuner._testing.testcase import DeterministicDDPTestCase
from xtuner.v1.utils.misc import monkey_patch_hf_modules_cache
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.loss.moe_loss import BalancingLossConfig, ZLossConfig
from xtuner.v1.model.base import BaseModel, HFSaveCfg, ModelItem, XTunerBaseModelConfig
from xtuner.v1.model.dense.qwen3 import Qwen3DenseConfig
from xtuner.v1.model.moe.moe import MoE, MoEConfig, SequenceContext
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.router import NoAuxRouterConfig


class _ReduceSumDDPTestCase(DeterministicDDPTestCase):
    """DDP test base for reduce-sum tests.

    These are gradient-PARITY tests that assert with tolerances, not bitwise reproducibility.
    ``enable_full_determinism`` (``torch.use_deterministic_algorithms``) NaNs the MoE EP backward on
    this base, so it is skipped here; the project's own EP parity scratchpad likewise runs without
    it. Everything else (hf module cache patch, prepare) is preserved.
    """

    def run_func(self, test_name):
        monkey_patch_hf_modules_cache()
        self.prepare()
        return getattr(self, test_name)()


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


def _tiny_moe_config(
    ep_size: int,
    balancing: bool,
    z_loss: bool,
    mtp: bool = False,
    dispatcher: str | None = None,
    router_bias_update_speed: float = 0.001,
) -> MoEConfig:
    # hidden_size / moe_intermediate_size must be >= 512: the EP grouped-gemm kernel JIT-compiled by
    # the all2all/DeepEP dispatcher fails ("PassManager::run failed") on smaller shapes. ep>1 uses the
    # all2all dispatcher (naive is ep=1 only); ep=1 keeps the default (naive) dispatcher.
    resolved_dispatcher = dispatcher if dispatcher is not None else ("all2all" if ep_size > 1 else None)
    return MoEConfig(
        vocab_size=1024,
        max_position_embeddings=512,
        pad_token_id=0,
        eos_token_id=0,
        num_hidden_layers=2,
        hidden_size=512,
        intermediate_size=1024,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        hidden_act="silu",
        attention=MHAConfig(num_attention_heads=16, num_key_value_heads=16, head_dim=32),
        tie_word_embeddings=False,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        hidden_factor=1.0,
        moe_intermediate_size=512,
        compile_cfg=False,
        router=NoAuxRouterConfig(
            scoring_func="sigmoid",
            router_scaling_factor=1.0,
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            router_bias_update_speed=router_bias_update_speed,
        ),
        ep_size=ep_size,
        dispatcher=resolved_dispatcher,
        balancing_loss_cfg=BalancingLossConfig() if balancing else None,
        z_loss_cfg=ZLossConfig() if z_loss else None,
        mtp_config=MTPConfig(num_layers=1, loss_scaling_factor=1.0) if mtp else None,
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


class TestReduceSumEndToEnd(_ReduceSumDDPTestCase):
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

    def _run_display_step(self, per_rank_seed: bool):
        # Build a tiny MoE (balancing + z on), run one train_step, and return this rank's displayed
        # losses. `per_rank_seed` chooses distinct (True) or identical (False) data across ranks.
        device = "cuda"
        rank = dist.get_rank()
        config = _tiny_moe_config(ep_size=1, balancing=True, z_loss=True)
        torch.manual_seed(0)
        engine = TrainEngine(
            model_cfg=config, optim_cfg=AdamWConfig(), fsdp_cfg=FSDPConfig(cpu_offload=False, reduce_dtype=torch.bfloat16)
        )
        engine.init_model_weights()
        gen = torch.Generator().manual_seed(1234 + (rank if per_rank_seed else 0))
        ids = torch.randint(0, 512, (1, 49), generator=gen).to(device)
        seq_ctx = SequenceContext.from_input_ids(input_ids=(ids[:, :-1],), device=device)
        lcs = engine.model.build_loss_ctx_batch([{"seq_ctx": seq_ctx, "shifted_labels": ids[:, 1:]}], sp_mesh=None)
        info = engine.train_step([ModelItem(seq_ctx=seq_ctx, loss_ctx=lcs[0])])
        return info

    def test_display_loss_is_per_rank_no_all_reduce(self):
        # Each rank must display ITS OWN loss (calibrate() -> per-token / per-rank mean, NO cross-rank
        # all_reduce). Distinct data => the displayed values differ across ranks; identical data =>
        # every rank shows the same value (which, being the whole batch on each rank, is the global
        # per-token mean, i.e. the world-size-1 value).
        self.create_pg("cuda")
        rank = dist.get_rank()
        world = dist.get_world_size()

        info = self._run_display_step(per_rank_seed=True)
        vals = [torch.zeros((), device="cuda") for _ in range(world)]
        dist.all_gather(vals, torch.tensor(info["logs_info"]["reduced_llm_loss"], device="cuda"))
        if rank == 0:
            assert torch.isfinite(torch.stack(vals)).all(), "non-finite per-rank display loss"
            assert (vals[0] - vals[1]).abs() > 1e-3, "distinct-data per-rank display losses are identical (all_reduced?)"

        info_sym = self._run_display_step(per_rank_seed=False)
        vals_sym = [torch.zeros((), device="cuda") for _ in range(world)]
        dist.all_gather(vals_sym, torch.tensor(info_sym["logs_info"]["reduced_llm_loss"], device="cuda"))
        if rank == 0:
            # Identical data on every rank => each rank's per-token mean equals the global one (up to
            # bf16 cross-rank forward noise, since the base runs non-deterministically).
            rel = ((vals_sym[0] - vals_sym[1]).abs() / vals_sym[0].abs().clamp_min(1e-6)).item()
            assert rel < 0.02, f"identical-data per-rank display losses differ by {rel:.4f} (>2%)"
            assert 0.0 < vals_sym[0].item() < 20.0, f"unreasonable per-token-mean CE {vals_sym[0].item()}"

    def test_reduce_sum_ep2_balancing_integration(self):
        # EP=2 + balancing integration. The router gate sits on a Replicate placement over the EP
        # sub-mesh; balancing loss flows ONLY into the gate (tokens_per_expert is detached, so every
        # other param receives only CE gradient). Verify (a) all non-gate params still match the
        # token-mean CE reference tightly -- i.e. EP=2 + balancing does not corrupt the main gradient
        # flow -- and (b) the router gate gets a finite, non-zero gradient. The exact correctness of
        # the gate's replicate-group SUM reduction with balancing is proven element-wise by
        # TestBalancingLossReduceSum (that reduce group is the same kind as this EP sub-mesh); a tight
        # dist-vs-single-process gate ratio is unreliable here because the ep-replicated
        # tokens_per_expert_global statistics differ from an ep=1 reference by construction.
        self.create_pg("cuda")
        device = "cuda"
        rank = dist.get_rank()
        ep_size = 2
        n_microbatch = 2
        seq_len = 32
        efsdp = self.world_size // ep_size

        config = _tiny_moe_config(ep_size=ep_size, balancing=True, z_loss=False)
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

        gen = torch.Generator().manual_seed(1234)
        shards = [torch.randint(0, 512, (1, seq_len + 1), generator=gen) for _ in range(efsdp * n_microbatch)]

        def build_loss_data(ids_list):
            out = []
            for ids in ids_list:
                ids = ids.to(device)
                seq_ctx = SequenceContext.from_input_ids(input_ids=(ids[:, :-1],), device=device)
                out.append({"seq_ctx": seq_ctx, "shifted_labels": ids[:, 1:]})
            return out

        fsdp_idx = rank // ep_size
        my_loss_data = build_loss_data(shards[fsdp_idx * n_microbatch : (fsdp_idx + 1) * n_microbatch])
        my_lcs = engine.model.build_loss_ctx_batch(my_loss_data, sp_mesh=None)
        items = [ModelItem(seq_ctx=d["seq_ctx"], loss_ctx=lc) for d, lc in zip(my_loss_data, my_lcs)]
        engine.model.zero_grad(set_to_none=True)
        engine.train_step(items)
        engine.model.scale_and_reduce_grad()
        dist_grads = _full_grads(engine.model)

        # Router gate must receive a finite, non-zero gradient on every rank (balancing reaches it).
        gate_names = [n for n in dist_grads if n.endswith(".gate.weight")]
        assert gate_names, "no router gate params found"
        for n in gate_names:
            assert torch.isfinite(dist_grads[n]).all(), f"non-finite gate gradient {n}"
            assert dist_grads[n].abs().sum() > 0, f"gate gradient {n} is all zero"

        if rank == 0:
            # Reference: CE-only, ep=1, full batch. Non-gate params receive only CE gradient, so they
            # must match tightly even with balancing enabled on the distributed side.
            ref = MoE(config=_tiny_moe_config(ep_size=1, balancing=False, z_loss=False)).to(torch.bfloat16).to(device)
            _, unexpected = ref.load_state_dict(
                {k: v.to(torch.bfloat16).to(device) for k, v in gold_weights.items()}, strict=False
            )
            assert not unexpected, f"unexpected keys: {unexpected}"
            ref.zero_grad(set_to_none=True)
            ref_loss_data = build_loss_data(shards)
            loss_cfg = CELossConfig()
            with _fake_dist_uninitialized():
                ref_lcs = loss_cfg.loss_ctx_cls.build_batches(
                    [loss_cfg.build(data={"shifted_labels": d["shifted_labels"]}, sp_mesh=None) for d in ref_loss_data]
                )
                total = torch.zeros((), device=device)
                for d, lc in zip(ref_loss_data, ref_lcs):
                    total = total + ref(seq_ctx=d["seq_ctx"], loss_ctx={"lm": lc})["loss"]
                total.backward()
            ref_grads = {n: p.grad.detach().float().cpu() for n, p in ref.named_parameters() if p.grad is not None}

            ratios = []
            for name, rg in ref_grads.items():
                if name.endswith(".gate.weight"):  # gate carries the balancing contribution; checked above
                    continue
                dg = dist_grads.get(name)
                if dg is None or dg.shape != rg.shape:
                    continue
                ratios.append((dg.norm() / rg.norm().clamp_min(1e-12)).item())
            ratios_t = torch.tensor(ratios)
            median = ratios_t.median().item()
            assert abs(median - 1.0) < 0.05, f"ep2+balancing non-gate grad ratio median={median} (want ~1.0)"
            assert (ratios_t - 1.0).abs().max().item() < 0.3, f"ep2+balancing a non-gate grad drifted: {ratios_t}"

    def test_mtp_micro_batch_forward_runs(self):
        # Guards the domino-EP MTP path in _micro_batch_forward, which calls aux_loss.accumulate for
        # the MTP routed experts. Before the world_size cleanup fix, that call still passed the removed
        # world_size arg (referencing a deleted z_world_size) and raised NameError/TypeError. The
        # domino micro-batch path needs ep>1 + a real dispatcher (naive is ep=1 only); exercise it
        # with mtp_config + intra_layer_micro_batch>1 + balancing/z on. It must run and log mtp loss.
        self.create_pg("cuda")
        device = "cuda"
        seq_len = 32

        # router_bias_update_speed=0 disables NoAuxRouter's update_bias, which is unrelated to this
        # path and crashes on the extra MTP row that the domino block adds to tokens_per_expert
        # (a pre-existing base issue, not part of the reduce-sum change under test here).
        config = _tiny_moe_config(
            ep_size=2, balancing=True, z_loss=True, mtp=True, dispatcher="all2all", router_bias_update_speed=0.0
        )
        torch.manual_seed(0)
        fsdp_cfg = FSDPConfig(cpu_offload=False, ep_size=2, reduce_dtype=torch.bfloat16)
        engine = TrainEngine(
            model_cfg=config, optim_cfg=AdamWConfig(), fsdp_cfg=fsdp_cfg, intra_layer_micro_batch=2
        )
        engine.init_model_weights()

        # ep replicas (world=2, ep=2 -> one fsdp position) must consume identical data.
        gen = torch.Generator().manual_seed(1234)
        loss_data = []
        for _ in range(2):
            ids = torch.randint(0, 512, (1, seq_len + 1), generator=gen).to(device)
            seq_ctx = SequenceContext.from_input_ids(input_ids=(ids[:, :-1],), device=device)
            loss_data.append({"seq_ctx": seq_ctx, "shifted_labels": ids[:, 1:]})
        loss_ctx_list = engine.model.build_loss_ctx_batch(loss_data, sp_mesh=None)
        items = [ModelItem(seq_ctx=d["seq_ctx"], loss_ctx=lc) for d, lc in zip(loss_data, loss_ctx_list)]

        engine.model.zero_grad(set_to_none=True)
        info = engine.train_step(items)  # routes through _micro_batch_forward (domino-MTP accumulate)
        engine.model.scale_and_reduce_grad()

        assert torch.isfinite(torch.tensor(info["total_loss"])), "total_loss is not finite"
        assert "reduced_mtp_loss" in info["logs_info"], "MTP loss missing from logged curves"
        grads = _full_grads(engine.model)
        assert grads and all(torch.isfinite(g).all() for g in grads.values()), "non-finite MTP-path gradient"


class TestBalancingLossReduceSum(_ReduceSumDDPTestCase):
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
        loss_new = ctx.finalize(
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


class TestComposeReduceSumHook(_ReduceSumDDPTestCase):
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


class TestDenseFp32IgnoredParamReduce(_ReduceSumDDPTestCase):
    @property
    def world_size(self) -> int:
        return 2

    def test_fp32_ignored_param_grad_summed_across_ranks(self):
        # fp32 ignored_params (matched by fp32_keys_pattern) are Replicate DTensors excluded from FSDP,
        # so they get no reduce-scatter. Under reduce-sum, scale_and_reduce_grad must SUM their per-rank
        # local grads over the replicate group -- otherwise the replicated copies carry different grads
        # (diverge) and the effective gradient is local, not the global sum. Heterogeneous per-rank data
        # makes the missing reduction observable.
        self.create_pg("cuda")
        device = "cuda"
        rank = dist.get_rank()
        world = dist.get_world_size()
        fp32_name = "norm.weight"  # HF key model.norm.weight

        cfg = Qwen3DenseConfig(
            vocab_size=1024,
            max_position_embeddings=512,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            num_hidden_layers=2,
            hidden_size=256,
            intermediate_size=512,
            rms_norm_eps=1e-6,
            rope_theta=1e6,
            hidden_act="silu",
            attention=MHAConfig(num_attention_heads=8, num_key_value_heads=8, head_dim=32),
            tie_word_embeddings=False,
            compile_cfg=False,
            hf_save_cfg=HFSaveCfg(fp32_keys_pattern=[r"model\.norm\.weight"]),
        )
        torch.manual_seed(0)
        engine = TrainEngine(
            model_cfg=cfg, optim_cfg=AdamWConfig(), fsdp_cfg=FSDPConfig(cpu_offload=False, reduce_dtype=torch.bfloat16)
        )
        engine.init_model_weights()

        p = dict(engine.model.named_parameters())[fp32_name]
        assert p.dtype == torch.float32, "fp32_keys_pattern param should stay fp32"
        assert isinstance(p, DTensor) and any(isinstance(pl, Replicate) for pl in p.placements), (
            "fp32 ignored param should be a Replicate DTensor excluded from FSDP"
        )

        gen = torch.Generator().manual_seed(1234)
        seqs = [torch.randint(0, 512, (1, 33), generator=gen) for _ in range(world)]
        ids = seqs[rank].to(device)
        seq_ctx = SequenceContext.from_input_ids(input_ids=(ids[:, :-1],), device=device)
        loss_cfg = CELossConfig()
        lc = loss_cfg.loss_ctx_cls.build_batches([loss_cfg.build(data={"shifted_labels": ids[:, 1:]}, sp_mesh=None)])[0]
        engine.model.zero_grad(set_to_none=True)
        engine.train_step([ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": lc})])

        def local(g):
            return (g.to_local() if isinstance(g, DTensor) else g).detach().float()

        # Pre-reduction per-rank local grad; the correct reduced value is the SUM over ranks.
        pre = local(p.grad).clone()
        gathered = [torch.zeros_like(pre) for _ in range(world)]
        dist.all_gather(gathered, pre.contiguous())
        expected_sum = sum(gathered)
        if rank == 0:
            assert (gathered[0] - gathered[1]).abs().max() > 1e-6, "per-rank grads identical; test is vacuous"

        # A FSDP-sharded (Shard placement) param, to co-test alongside the Replicate one. Its exact
        # reduce-sum value (== single-process global-batch gradient) is covered by the EP1/EP2
        # token-mean parity tests; here we only confirm the Shard path produces a finite, non-zero,
        # cross-rank-consistent gradient (i.e. the reduce-scatter ran and did not corrupt it).
        shard_name = "embed_tokens.weight"
        sp = dict(engine.model.named_parameters())[shard_name]
        assert isinstance(sp, DTensor) and any(isinstance(pl, Shard) for pl in sp.placements), (
            "expected a Shard-placement param to co-test with the Replicate one"
        )
        shard_full = sp.grad.full_tensor().detach().float()
        assert torch.isfinite(shard_full).all() and shard_full.abs().sum() > 0, "shard-param grad invalid"

        engine.model.scale_and_reduce_grad()
        post = local(p.grad)

        # Replicate param: every rank's reduced grad equals the global SUM (and is therefore consistent).
        torch.testing.assert_close(post, expected_sum, rtol=1e-3, atol=1e-3)
        allpost = [torch.zeros_like(post) for _ in range(world)]
        dist.all_gather(allpost, post.contiguous())
        shard_all = [torch.zeros_like(shard_full) for _ in range(world)]
        dist.all_gather(shard_all, shard_full.contiguous())
        if rank == 0:
            assert (allpost[0] - allpost[1]).abs().max() < 1e-4, "reduced fp32 (Replicate) grad not consistent"
            assert (shard_all[0] - shard_all[1]).abs().max() < 1e-4, "Shard-param full grad not consistent"
