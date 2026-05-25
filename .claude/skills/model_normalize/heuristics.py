"""Default heuristic rule set for FP8 quantization without a reference model.

Rule of thumb
-------------
Quantize the *large matrix multiplications* in the transformer block; keep
everything else in the original dtype. Concretely this means:

QUANTIZE
    * ``self_attn.{q,k,v,o}_proj.weight`` (standard attention)
    * ``mlp.{gate,up,down}_proj.weight``  (dense MLP)
    * ``mlp.experts.<i>.{gate,up,down}_proj.weight`` (per-expert MoE)
    * ``mlp.experts.gate_up_proj`` / ``mlp.experts.down_proj``
      (fused 3D MoE-expert weights)
    * ``mlp.shared_expert.{gate,up,down}_proj.weight`` (MoE shared expert)
    * ``linear_attn.{in_proj_qkv, in_proj_z, out_proj}.weight``
      (the wide projections in hybrid / Mamba-style blocks)

  These patterns apply both under ``model.language_model.layers.<i>.`` and
  ``mtp.layers.<i>.`` (multi-token-prediction blocks share the language
  model's block layout).

KEEP IN ORIGINAL DTYPE (excluded by construction — no negative regex needed)
    * all norms (``*norm*.weight``, ``*_layernorm.weight``, ``norm.bias``)
    * MoE routers (``mlp.gate.weight``, ``mlp.shared_expert_gate.weight``)
    * embeddings (``embed_tokens``, ``lm_head``, ``pos_embed``, ``patch_embed``, ``mtp.fc``)
    * vision tower (``model.visual.*``)
    * all ``*.bias`` tensors
    * ``linear_attn`` control-flow tensors: ``A_log``, ``conv1d.weight``,
      ``dt_bias``, ``in_proj_a.weight``, ``in_proj_b.weight``, ``norm.weight``

  These are kept because they are either small (negligible memory win),
  dtype-sensitive (norms / control-flow projections), or shape-incompatible
  with block-128 FP8 quantization (1D biases, embeddings).

This rule set was validated against the InternS2 / Qwen3-MoE FP8 reference
layouts produced by SGLang / vLLM tooling. If the source model uses different
naming for any of these concepts, fall back to the ``--reference`` mode.
"""

from __future__ import annotations

from typing import Callable

from hf_to_fp8 import compile_union

DEFAULT_QUANTIZE_PATTERNS: list[str] = [
    # ---- Standard self-attention projections ---------------------------------
    r"^(?:model\.)?(?:language_model\.)?layers\.\d+\.self_attn\.[qkvo]_proj\.weight$",
    r"^mtp\.layers\.\d+\.self_attn\.[qkvo]_proj\.weight$",
    # ---- Dense MLP (non-MoE) -------------------------------------------------
    r"^(?:model\.)?(?:language_model\.)?layers\.\d+\.mlp\.(?:gate|up|down)_proj\.weight$",
    # ---- MoE per-expert linears (unfused) -----------------------------------
    r"^(?:model\.)?(?:language_model\.)?layers\.\d+\.mlp\.experts\.\d+\.(?:gate|up|down)_proj\.weight$",
    r"^mtp\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate|up|down)_proj\.weight$",
    # ---- MoE fused-expert linears (3D, single tensor per layer) -------------
    # Note: no trailing ``.weight`` — these names come from the fused
    # representation used by vLLM / SGLang fused MoE kernels.
    r"^(?:model\.)?(?:language_model\.)?layers\.\d+\.mlp\.experts\.(?:gate_up_proj|down_proj)$",
    # ---- MoE shared expert (always-on dense path) ---------------------------
    r"^(?:model\.)?(?:language_model\.)?layers\.\d+\.mlp\.shared_expert\.(?:gate|up|down)_proj\.weight$",
    r"^mtp\.layers\.\d+\.mlp\.shared_expert\.(?:gate|up|down)_proj\.weight$",
    # ---- Linear-attention wide projections ----------------------------------
    r"^(?:model\.)?(?:language_model\.)?layers\.\d+\.linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)\.weight$",
    r"^mtp\.layers\.\d+\.linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)\.weight$",
]


def build_heuristic_predicate() -> Callable[[str], bool]:
    """Return a predicate matching ``DEFAULT_QUANTIZE_PATTERNS``.

    Returns:
        Callable[[str], bool]: ``True`` iff the tensor name should be FP8-quantized.
    """
    pattern = compile_union(DEFAULT_QUANTIZE_PATTERNS)
    return lambda name: pattern.search(name) is not None
