# Copyright (c) OpenMMLab. All rights reserved.
# Hash routing semantics ported from DeepSeek-V4-Flash inference/model.py (MIT) and
# the HF transformers v5.8.1 DeepseekV4HashGate (Apache-2.0); see
# docs/design/deepseek_v4_support.md §4.6.
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict

from .protocol import RouterProtocol, RouterResults


class HashRouterConfig(BaseModel):
    """Config for :class:`HashRouter`.

    The first ``num_hash_layers`` MoE layers of DeepSeek-V4-Flash route tokens to experts
    deterministically by token id via a lookup table ``tid2eid``; no gate logits are computed.
    """

    model_config = ConfigDict(extra="forbid")
    vocab_size: int
    n_routed_experts: int
    num_experts_per_tok: int

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
        )


class HashRouter(nn.Module, RouterProtocol):
    """Hash-routing gate for DeepSeek-V4-Flash early MoE layers.

    Uses an int32 ``tid2eid: [vocab_size, num_experts_per_tok]`` buffer (loaded from
    checkpoint, NOT trained) to map each token id to its activated experts. No gate
    logits are computed; ``topk_weights`` is uniform ``1 / num_experts_per_tok``.

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
    """

    tid2eid: torch.Tensor

    def __init__(
        self,
        *,
        vocab_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_routed_experts = n_routed_experts
        self.top_k = num_experts_per_tok
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
        # `logits` and `rollout_routed_experts` exist for protocol compatibility but are
        # ignored: the hash branch never computes scores and cannot be retro-routed.
        del logits, rollout_routed_experts
        assert input_ids is not None, (
            "HashRouter requires `input_ids` to be passed via the MoEDecoderLayer input_ids kwarg; got None."
        )

        # Flatten so output shape matches the 2D `[total_tokens, ...]` contract used by
        # score-based routers (NoAux / Greedy), regardless of whether the caller passed
        # `[B, S]` or already-packed `[total_tokens]`.
        flat_input_ids = input_ids.reshape(-1).long()
        # Lookup per-token expert assignment. tid2eid is int32; cast result to long so
        # downstream `torch.histc` / `gather` ops match the dtype contract of other routers.
        topk_ids = self.tid2eid[flat_input_ids].long()
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32) / self.top_k

        # `torch.histc` does not support integer tensors on CPU; cast to float so the call
        # works on both CUDA (where Long is supported) and CPU. This mirrors the integer
        # bin count behavior used by score-based routers.
        tokens_per_expert = torch.histc(
            topk_ids.to(torch.float32),
            bins=self.n_routed_experts,
            min=0,
            max=self.n_routed_experts,
        )

        # `RouterResults.logits` is typed as `torch.Tensor` (TypedDict invariant); since
        # hash routing produces no logits we emit a tiny zero tensor on the right device.
        # Downstream consumers (`MoEDecoderLayer._forward` returns it, `aux_loss.accumulate`
        # concatenates it) would crash on None. PR9 will gate aux-loss/z-loss off for
        # hash layers, at which point this dummy can be revisited.
        dummy_logits = torch.zeros(1, device=topk_ids.device, dtype=torch.float32)
        # `router_weights` (used by aux balance loss) is similarly meaningless for hash
        # routing; emit a uniform distribution per token so any accidental aux-loss path
        # remains numerically benign instead of NaN.
        router_weights = torch.full(
            (topk_ids.shape[0], self.n_routed_experts),
            1.0 / self.n_routed_experts,
            device=topk_ids.device,
            dtype=torch.float32,
        )

        return {
            "logits": dummy_logits,
            "router_weights": router_weights,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "topkens_per_expert": tokens_per_expert,
        }
