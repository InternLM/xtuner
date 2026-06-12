# Copyright (c) OpenMMLab. All rights reserved.
"""Muon optimizer tests.

1. TestNewtonSchulz     — Triton vs PyTorch Newton-Schulz precision.
2. TestMuonSingleGPU   — Single-GPU correctness: production Muon must match the
                          reference within tight tolerances (fp32, no comms).
3. TestMuonFSDP        — End-to-end correctness: one Muon step on a fully-sharded
                          model must match a single-process reference for every
                          parameter, covering all four communication strategies
                          (all_to_all, agrs, local, subgroup_allgather).
"""

import copy
import math
from typing import Callable, overload

import parametrize
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from xtuner._testing.testcase import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.config.optim import MuonConfig
from xtuner.v1.model.base import BaseModel, XTunerBaseModelConfig
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseMLP
from xtuner.v1.optim.muon import zeropower_via_newtonschulz5


# ─── Test: Newton-Schulz functions ───────────────────────────────────────────


class TestNewtonSchulz(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 1

    def test_triton_vs_pytorch(self):
        """Triton and PyTorch NS implementations should produce similar results."""
        self.create_pg("cuda")
        from xtuner.v1.optim.newton_schulz_triton import newton_schulz_triton

        # Use a single representative shape per category to limit autotune overhead.
        # Autotune results are cached on disk, so subsequent runs are fast.
        test_cases = [
            (1, 512, 1536),  # regular matrix (M < N)
            (4, 256, 256),  # MoE experts (square)
        ]

        for num_experts, M, N in test_cases:
            G = torch.randn(num_experts * M, N, device="cuda", dtype=torch.float32)

            result1 = zeropower_via_newtonschulz5(G, epsilon=1e-7, num_experts=num_experts)
            result2 = newton_schulz_triton(G, epsilon=1e-7, num_experts=num_experts)

            assert not torch.isnan(result1).any()
            assert not torch.isnan(result2).any()
            torch.testing.assert_close(result1, result2, atol=3e-2, rtol=3e-2)


# ─── Model for end-to-end tests ──────────────────────────────────────────────


class ToyMoEModelConfig(XTunerBaseModelConfig):
    vocab_size: int = 64
    hidden_size: int = 32
    intermediate_size: int = 64
    n_routed_experts: int = 2
    moe_intermediate_size: int = 16

    def build(self) -> "ToyMoEModel":
        return ToyMoEModel(self)


class ToyMoEModel(BaseModel):
    """Minimal model exercising all Muon param categories and comm strategies.

    Forward skips top-k routing for simplicity — all experts are applied
    unconditionally, guaranteeing every expert weight receives non-zero
    gradients so the optimizer update is exercised on all of them.

    With world_size=4 and n_routed_experts=2, the comm strategy coverage is:
      all_to_all          — fcs (4 same-shape params fill a full batch)
      agrs                — mlp params (remainder batches, grouped by shape)
      local               — fused_w1w3 (ns_experts=4, 4%4==0, 1 expert/rank)
      subgroup_allgather  — fused_w2 (ns_experts=2, 4%2==0, sg_size=2)

    Parameter layout:
      embed_tokens.weight       (64, 32)  — AdamW (name-matched)
      fcs.{0,1,2,3}.weight      (32, 32)  — Muon regular
      mlp.gate_proj.weight      (64, 32)  — Muon regular
      mlp.up_proj.weight        (64, 32)  — Muon regular
      mlp.down_proj.weight      (32, 64)  — Muon regular
      norm.weight               (32,)     — AdamW (1D)
      norm.bias                 (32,)     — AdamW (1D)
      fused_w1w3.weight         (64, 32)  — Muon MoE, num_experts = 2 * 2 = 4
      fused_w2.weight           (64, 16)  — Muon MoE, num_experts = 2
      lm_head.weight            (64, 32)  — AdamW (name-matched)
    """

    config: ToyMoEModelConfig  # type: ignore

    def __init__(self, config: ToyMoEModelConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # 4 linear layers to fill a full batch (Muon regular, all_to_all)
        self.fcs = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size, bias=False) for _ in range(4)])
        # DenseMLP — 3 more Muon regular params (remainder batch, agrs)
        self.mlp = DenseMLP(
            hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, hidden_act="silu"
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        # fused_w1w3: (n_routed_experts * 2 * moe_intermediate_size, hidden_size)
        self.fused_w1w3 = nn.Linear(
            config.hidden_size, config.n_routed_experts * 2 * config.moe_intermediate_size, bias=False
        )
        # fused_w2: (n_routed_experts * hidden_size, moe_intermediate_size)
        self.fused_w2 = nn.Linear(
            config.moe_intermediate_size, config.n_routed_experts * config.hidden_size, bias=False
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Trivial forward that touches all params and returns a scalar loss."""
        h = self.embed_tokens(input_ids)
        h = h + sum(fc(h) for fc in self.fcs)
        h = self.norm(h)
        h = h + self.mlp(h)
        # Simplified MoE: split → SiLU gate → w2 (no routing, all experts applied)
        w1w3_out = self.fused_w1w3(h)
        gate, up = w1w3_out.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        # Per-expert w2: reshape to (*, n_experts, intermediate) then batch matmul
        hidden = hidden.view(*hidden.shape[:-1], self.config.n_routed_experts, self.config.moe_intermediate_size)
        w2 = self.fused_w2.weight.view(
            self.config.n_routed_experts, self.config.hidden_size, self.config.moe_intermediate_size
        )
        expert_out = torch.einsum("...ei,ehi->...eh", hidden, w2).sum(-2)
        h = h + expert_out
        logits = self.lm_head(h)
        return logits.sum()

    @overload  # type:ignore
    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor: ...  # type:ignore

    __call__ = nn.Module.__call__


# ─── Reference optimizer ─────────────────────────────────────────────────────


class ReferenceMuon(torch.optim.Optimizer):
    """Self-contained single-process reference for Muon + AdamW.

    Classifies parameters using the same name-based rules as MuonConfig.build():
    - 2D+ tensors (excluding embed_tokens, lm_head, effectively-1D) → Muon
    - Everything else → AdamW
    """

    FUSED_W1W3_PATTERNS = ("fused_w1w3",)
    OTHER_EXPERT_PATTERNS = ("fused_w2", "fused_w1", "fused_w3")

    NS_CONSTS = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    def __init__(
        self,
        model: ToyMoEModel,
        lr: float,
        mu: float,
        weight_decay: float,
        epsilon: float,
        betas: tuple[float, float] = (0.9, 0.95),
    ):
        n_routed_experts = model.config.n_routed_experts
        muon_groups: list[dict] = []
        adamw_params: list[nn.Parameter] = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_muon = p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name and p.numel() not in p.shape
            if is_muon:
                ne = self._get_num_experts(name, n_routed_experts)
                muon_groups.append(
                    {
                        "params": [p],
                        "type": "muon",
                        "num_experts": ne,
                        "base_lr": lr,
                        "lr": lr * self._rms_lr_ratio(p, ne),
                    }
                )
            else:
                adamw_params.append(p)

        super().__init__(
            muon_groups + [{"params": adamw_params, "type": "adamw", "base_lr": lr, "lr": lr}],
            {"mu": mu, "weight_decay": weight_decay, "epsilon": epsilon, "betas": betas},
        )

    @staticmethod
    def _rms_lr_ratio(p: nn.Parameter, num_experts: int) -> float:
        A = p.shape[-2] // num_experts
        B = p.shape[-1]
        return 0.2 * math.sqrt(max(A, B))

    @classmethod
    def _get_num_experts(cls, name: str, n_routed_experts: int) -> int:
        if any(pat in name for pat in cls.FUSED_W1W3_PATTERNS):
            return 2 * n_routed_experts
        elif any(pat in name for pat in cls.OTHER_EXPERT_PATTERNS):
            return n_routed_experts
        return 1

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["mu"]
            wd = group["weight_decay"]
            eps = group["epsilon"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if group["type"] == "muon":
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p.data, dtype=torch.float32)

                    mom = state["momentum"]
                    num_experts = group["num_experts"]
                    g = p.grad.to(dtype=torch.float32)

                    # Momentum + Nesterov
                    mom.mul_(mu).add_(g)
                    U = mom * mu + g

                    # Newton-Schulz orthogonalization
                    X = U.to(dtype=torch.bfloat16)
                    N = X.size(-1)
                    X = X.view(num_experts, -1, N)
                    need_transpose = X.size(-2) > X.size(-1)
                    if need_transpose:
                        X = X.mT
                    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
                    for a, b, c in self.NS_CONSTS:
                        A = X @ X.mT
                        B = b * A + c * (A @ A)
                        X = a * X + B @ X
                    if need_transpose:
                        X = X.mT
                    X = X.view(U.shape)

                    # Weight decay (uses base_lr, not adjusted_lr)
                    base_lr = group["base_lr"]
                    p.data.mul_(1 - base_lr * wd)
                    # Update: multiply in bf16 (matches production _foreach_mul_ on bf16)
                    X.mul_(lr)
                    p.data.sub_(X)

                else:  # AdamW
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32)
                        state["step"] = 0

                    state["step"] += 1
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    g = p.grad.float()

                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]
                    m_hat = m / bc1
                    v_hat = v / bc2

                    # Decoupled weight decay
                    p.data.mul_(1 - lr * wd)
                    p.data.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)


# ─── Test: single-GPU correctness ───────────────────────────────────────────


class TestMuonSingleGPU(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 1

    def test_muon_single_gpu_matches_reference(self):
        """On a single GPU, production Muon must match the reference for every
        parameter within tight tolerances (everything is fp32, no comms).

        fully_shard() is required even here: the production Muon optimizer
        assumes that parameters are DTensors and will fail on plain tensors.
        With world_size=1 it is a no-op sharding-wise.
        """
        self.create_pg("cuda")
        device = "cuda"

        LR = 0.01
        MU = 0.95
        WD = 0.01
        EPSILON = 1e-8
        BETAS = (0.9, 0.95)

        # ── Build two identical models ───────────────────────────────────────
        torch.manual_seed(42)
        config = ToyMoEModelConfig(compile_cfg=False)
        model = config.build().to(device)
        ref_model = copy.deepcopy(model)
        input_ids = torch.randint(0, config.vocab_size, (2, 8), device=device)

        # ── Reference path ───────────────────────────────────────────────────
        ref_loss = ref_model(input_ids)
        ref_loss.backward()
        ref_optim = ReferenceMuon(ref_model, lr=LR, mu=MU, weight_decay=WD, epsilon=EPSILON, betas=BETAS)
        ref_optim.step()

        # ── Production path ──────────────────────────────────────────────────
        fsdp_config = FSDPConfig(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            torch_compile=False,
        )
        model.fully_shard(fsdp_config)

        loss = model(input_ids)
        loss.backward()
        muon_config = MuonConfig(lr=LR, momentum=MU, weight_decay=WD, eps=EPSILON, betas=BETAS)
        optim = muon_config.build(model)
        optim.step()

        # ── Compare all parameters ───────────────────────────────────────────
        for (name, ref_p), (_, prod_p) in zip(ref_model.named_parameters(), model.named_parameters()):
            full = prod_p.data.full_tensor()  # type: ignore[attr-defined]
            abs_diff = (full - ref_p.data).abs()
            rel_diff = abs_diff / (ref_p.data.abs().clamp(min=1e-12))
            torch.testing.assert_close(
                full,
                ref_p.data,
                atol=1e-6,
                rtol=1e-5,
                msg=f"mismatch on '{name}': max_abs={abs_diff.max().item():.2e}, max_rel={rel_diff.max().item():.2e}",
            )


# ─── Test: end-to-end FSDP ───────────────────────────────────────────────────


class TestMuonFSDP(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 4

    @parametrize.parametrize("enable_all2all", [True, False])
    def test_muon_fsdp_matches_reference(self, enable_all2all: bool):
        """One Muon step on a fully-sharded model must match the single-process
        reference for every parameter, across all param categories.

        Each rank processes a different sample (as in real DDP/FSDP training).
        The reference processes all samples and averages the loss to produce
        equivalent gradients.
        """
        self.create_pg("cuda")
        device = "cuda"
        rank = dist.get_rank()
        ws = dist.get_world_size()

        LR = 0.01
        MU = 0.95
        WD = 0.01
        EPSILON = 1e-8
        BETAS = (0.9, 0.95)

        # ── Build model on every rank, then broadcast rank-0 weights ─────────
        config = ToyMoEModelConfig(compile_cfg=False)
        model = config.build().to(device)
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

        ref_model = copy.deepcopy(model)

        # ── Generate input data on rank 0 and scatter ────────────────────────
        if rank == 0:
            torch.manual_seed(7)
            all_input_ids = torch.randint(0, config.vocab_size, (ws, 8), device=device)
        else:
            all_input_ids = torch.empty(ws, 8, dtype=torch.long, device=device)
        dist.broadcast(all_input_ids, src=0)

        # Each rank gets its own sample
        local_input_ids = all_input_ids[rank : rank + 1]

        # ── Reference: forward all samples, average loss ─────────────────────
        ref_loss = ref_model(all_input_ids) / ws
        ref_loss.backward()

        ref_optim = ReferenceMuon(ref_model, lr=LR, mu=MU, weight_decay=WD, epsilon=EPSILON, betas=BETAS)
        ref_optim.step()

        # ── Fully shard the test model (production path) ─────────────────────
        fsdp_config = FSDPConfig(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            torch_compile=False,
        )
        model.fully_shard(fsdp_config)

        # ── Forward + backward on sharded model (each rank sees its sample) ──
        fsdp_loss = model(local_input_ids)
        fsdp_loss.backward()

        # ── Production Muon optimizer step ────────────────────────────────────
        muon_config = MuonConfig(lr=LR, momentum=MU, weight_decay=WD, eps=EPSILON, betas=BETAS, enable_all2all=enable_all2all)
        optim = muon_config.build(model)
        optim.step()

        # ── Compare all parameters ───────────────────────────────────────────
        for (name, ref_p), (_, fsdp_p) in zip(ref_model.named_parameters(), model.named_parameters()):
            full = fsdp_p.data.full_tensor()  # type: ignore[attr-defined]
            abs_diff = (full - ref_p.data).abs()
            rel_diff = abs_diff / (ref_p.data.abs().clamp(min=1e-12))
            torch.testing.assert_close(
                full,
                ref_p.data,
                atol=1e-6,
                rtol=1e-5,
                msg=f"mismatch on '{name}': max_abs={abs_diff.max().item():.2e}, max_rel={rel_diff.max().item():.2e}",
            )
