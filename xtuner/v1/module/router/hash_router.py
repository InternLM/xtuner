# Copyright (c) OpenMMLab. All rights reserved.
# Hash routing semantics ported from DeepSeek-V4-Flash inference/model.py (MIT) and
# the HF transformers v5.8.1 DeepseekV4HashGate (Apache-2.0); see
# docs/design/deepseek_v4_support.md §4.6.
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict

from .protocol import RouterProtocol, RouterResults


class HashRouterConfig(BaseModel):
    """Config for :class:`HashRouter`.

    The first ``num_hash_layers`` MoE layers of DeepSeek-V4-Flash pick *which* experts a
    token goes to from a frozen ``tid2eid`` lookup instead of a score argmax. The scoring
    parameters below still apply: hash routing replaces only the selection step, and the
    per-expert combine weights come from the same learned gate as the score-routed layers.
    """

    model_config = ConfigDict(extra="forbid")
    vocab_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    scoring_func: Literal["sigmoid", "softmax", "sqrtsoftplus"]
    router_scaling_factor: float
    norm_topk_prob: bool = True

    def build(
        self,
        n_routed_experts: int,
        num_experts_per_tok: int,
    ) -> "HashRouter":
        # The MoE gate construction site (``MoEGate.__init__``) passes these for parity
        # with score-based router configs; for HashRouter they must match the config
        # fields, otherwise the ``tid2eid`` buffer shape and lookup are inconsistent.
        assert n_routed_experts == self.n_routed_experts, (
            f"HashRouterConfig.n_routed_experts={self.n_routed_experts} mismatches "
            f"MoEGate n_routed_experts={n_routed_experts}"
        )
        assert num_experts_per_tok == self.num_experts_per_tok, (
            f"HashRouterConfig.num_experts_per_tok={self.num_experts_per_tok} mismatches "
            f"MoEGate num_experts_per_tok={num_experts_per_tok}"
        )
        return HashRouter(
            vocab_size=self.vocab_size,
            n_routed_experts=self.n_routed_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            scoring_func=self.scoring_func,
            router_scaling_factor=self.router_scaling_factor,
            norm_topk_prob=self.norm_topk_prob,
        )


class HashRouter(nn.Module, RouterProtocol):
    """Hash-routing gate for DeepSeek-V4-Flash early MoE layers.

    Uses an int32 ``tid2eid: [vocab_size, num_experts_per_tok]`` buffer (loaded from
    checkpoint, NOT trained) to map each token id to its activated experts. Only the
    *selection* is frozen: the combine weights are the learned gate's scores gathered at
    those indices and renormalised, exactly as in :class:`NoAuxRouter` — see the V4
    reference ``Gate.forward``, where the ``hash`` branch changes ``indices`` only and
    falls through to the shared ``original_scores.gather(1, indices)`` tail.

    The ``input_ids`` argument to :meth:`forward` is REQUIRED for HashRouter
    (asserted, not optional). It is expected in packed ``[total_tokens]`` shape and
    must reach the gate from ``MoEDecoderLayer.forward`` via the ``input_ids``
    keyword that this PR threads through the MoE decoder layer.

    Args:
        vocab_size (int): Tokenizer vocabulary size; first dim of ``tid2eid``.
        n_routed_experts (int): Total number of routed experts; valid range of
            entries in ``tid2eid``.
        num_experts_per_tok (int): Number of experts each token is dispatched to;
            second dim of ``tid2eid``.
        scoring_func ("sigmoid" | "softmax" | "sqrtsoftplus"): Activation turning gate
            logits into expert scores.
        router_scaling_factor (float): Scalar applied to the final combine weights.
        norm_topk_prob (bool): Whether to renormalise the gathered weights to sum to 1.
    """

    tid2eid: torch.Tensor

    def __init__(
        self,
        *,
        vocab_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        scoring_func: Literal["sigmoid", "softmax", "sqrtsoftplus"],
        router_scaling_factor: float,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_routed_experts = n_routed_experts
        self.top_k = num_experts_per_tok
        self.scoring_func = scoring_func
        self.router_scaling_factor = router_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        # `tid2eid` is loaded from checkpoint (persistent buffer); the V4-Flash release
        # ships it alongside the gate weights, see HF DeepseekV4HashGate.
        self.register_buffer(
            "tid2eid",
            torch.zeros((vocab_size, num_experts_per_tok), dtype=torch.int32),
            persistent=True,
        )

    def forward(
        self,
        logits: torch.Tensor | None,
        rollout_routed_experts: torch.Tensor | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> RouterResults:
        # `rollout_routed_experts` exists for protocol compatibility but is ignored:
        # hash routing is already fully determined by the token id, so there is nothing
        # for a rollout to retro-route.
        del rollout_routed_experts
        assert input_ids is not None, (
            "HashRouter requires `input_ids` to be passed via the MoEDecoderLayer input_ids kwarg; got None."
        )
        assert logits is not None, "HashRouter needs the gate logits to weight the hashed experts; got None."

        match self.scoring_func:
            case "sigmoid":
                scores = logits.sigmoid()
            case "softmax":
                scores = logits.softmax(dim=-1)
            case "sqrtsoftplus":
                scores = F.softplus(logits).sqrt()
            case _:
                raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        # Flatten so output shape matches the 2D `[total_tokens, ...]` contract used by
        # score-based routers (NoAux / Greedy), regardless of whether the caller passed
        # `[B, S]` or already-packed `[total_tokens]`.
        flat_input_ids = input_ids.reshape(-1).long()
        # Lookup per-token expert assignment. tid2eid is int32; cast result to long so
        # downstream `torch.histc` / `gather` ops match the dtype contract of other routers.
        topk_ids = self.tid2eid[flat_input_ids].long()

        topk_weights = scores.gather(1, topk_ids)
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.router_scaling_factor

        # `torch.histc` does not support integer tensors on CPU; cast to float so the call
        # works on both CUDA (where Long is supported) and CPU. This mirrors the integer
        # bin count behavior used by score-based routers.
        tokens_per_expert = torch.histc(
            topk_ids.to(torch.float32),
            bins=self.n_routed_experts,
            min=0,
            max=self.n_routed_experts,
        )

        return {
            "logits": logits,
            "router_weights": scores / scores.sum(dim=-1, keepdim=True),
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "topkens_per_expert": tokens_per_expert,
        }
