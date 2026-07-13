# Copyright (c) OpenMMLab. All rights reserved.
"""DeepSeek-V4-Flash tests: checkpoint round-trip and forward parity vs HuggingFace.

Every test in this file drives the *production* HF path. A toy HF
:class:`~transformers.models.deepseek_v4.DeepseekV4ForCausalLM` is random-initialised at
tiny dimensions and written out as a DeepSeek-V4-Flash **release-format** checkpoint
(:func:`_export_release_checkpoint`); XTuner then consumes that directory through
``get_model_config_from_hf`` + ``build`` + ``from_hf``, exactly as it would a real
release. No test copies parameters module-by-module into an XTuner model, so the
key mapping in ``DeepSeekV4.to_hf_key_list`` is itself under test rather than bypassed.

Coverage:

* :meth:`TestDeepSeekV4Checkpoint.test_release_checkpoint_round_trip` — a one-layer model
  loaded with ``from_hf`` and written back with ``save_hf`` reproduces the source
  checkpoint's key set exactly and every tensor bitwise.
* :class:`TestDeepSeekV4Parity` — whole-model forward parity against the HF reference for
  each layer flavour in isolation (hash-routed MoE, CSA, HCA) and for stacked V4-Flash
  models, plus an exact check on the Indexer's chosen compressed entries.

There is no tolerance: every parity assertion is ``torch.equal``. That is possible
because the class runs under ``XTUNER_V4_HF_PARITY``
(:mod:`xtuner.v1.utils.hf_parity`), which puts each site where XTuner deliberately
computes something differently from HF — the HC bf16 fast paths, the fp32 gathered
attention core, the reference-faithful low-precision Indexer score — onto HF's exact op
sequence, shapes included. Those choices are all defensible in production and stay the
default; kernel-level numerics are covered by the op- and module-level tests. What these
tests own is *structure*: layer wiring, weight mapping, masking, routing, rope-base
selection. A tolerance is the wrong instrument for that — one wide enough to survive a
stacked model would also have passed every structural bug this file has caught.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


hf_v4 = pytest.importorskip("transformers.models.deepseek_v4")
from safetensors.torch import save_file
from transformers.models.deepseek_v4 import DeepseekV4Config as HFDeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.model.moe.deepseek_v4 import DeepSeekV4, DeepSeekV4Config
from xtuner.v1.utils.hf_parity import set_hf_parity


# ─── Toy model dimensions ───────────────────────────────────────────────────────
_VOCAB = 256
_HIDDEN = 64
_MOE_INTER = 32
_N_HEADS = 8
_HEAD_DIM = 32
_QK_ROPE = 16
_SLIDING = 32
_INDEX_TOPK = 8
_INDEX_HEAD_DIM = 16
# 4 index heads sits below the triton indexer kernel's tensor-core floor, which is why
# the XTuner side pins ``indexer_backend="native"`` for these fixtures.
_INDEX_N_HEADS = 4
_N_ROUTED = 4
_N_SHARED = 1
_N_EXPERTS_PER_TOK = 2
_O_GROUPS = 2
_O_LORA = 16
_Q_LORA = 32
_HC_MULT = 4
_HC_SINKHORN_ITERS = 3
_RMS_EPS = 1e-6
_DTYPE = torch.bfloat16
_MAX_POS = 2048
# YaRN, as in the release. ``original_max_position_embeddings`` is picked so the implicit
# factor (``max_position_embeddings / original``) equals the explicit one; transformers
# warns and silently prefers the explicit factor when they disagree, which would put the
# two sides on different rope bases.
_ROPE_SCALING = {
    "rope_type": "yarn",
    "factor": 16.0,
    "original_max_position_embeddings": _MAX_POS // 16,
    "beta_fast": 32,
    "beta_slow": 1,
}

_LAYER_TYPE_TO_RATIO = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}

_REAL_BF16_PATH_ENV = "DEEPSEEK_V4_BF16_PATH"
_REAL_BF16_DEFAULT = "/mnt/shared-storage-user/llmrazor-share/yehaochen/model/DeepSeek-V4-Flash"


# ─── HF → release-format checkpoint export ──────────────────────────────────────
#
# The HF implementation and the released DeepSeek-V4-Flash checkpoint use two different
# naming schemes for the same weights, and transformers ships no converter between them:
# HF nests the Indexer inside the CSA compressor, fuses the routed experts into
# ``gate_up_proj`` / ``down_proj``, and prefixes everything with ``model.``. The release
# (which XTuner's ``from_hf`` / ``save_hf`` target) keeps DeepSeek's own inference names.
# The two tables below are the single place those schemes meet.

_TOP_LEVEL_RENAMES = {
    "model.embed_tokens.weight": "embed.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "head.weight",
    "model.hc_head.hc_fn": "hc_head_fn",
    "model.hc_head.hc_base": "hc_head_base",
    "model.hc_head.hc_scale": "hc_head_scale",
}

# Applied to the tail after ``model.layers.<L>.``. Order matters: the nested indexer rules
# must run before the plain ``self_attn.compressor.`` ones they share a prefix with.
_LAYER_TAIL_RENAMES = [
    (r"^input_layernorm\.", "attn_norm."),
    (r"^post_attention_layernorm\.", "ffn_norm."),
    (r"^attn_hc\.", "hc_attn_"),
    (r"^ffn_hc\.", "hc_ffn_"),
    (r"^self_attn\.compressor\.indexer\.scorer\.weights_proj\.", "attn.indexer.weights_proj."),
    (r"^self_attn\.compressor\.indexer\.q_b_proj\.", "attn.indexer.wq_b."),
    (r"^self_attn\.compressor\.indexer\.kv_proj\.", "attn.indexer.compressor.wkv."),
    (r"^self_attn\.compressor\.indexer\.gate_proj\.", "attn.indexer.compressor.wgate."),
    (r"^self_attn\.compressor\.indexer\.kv_norm\.", "attn.indexer.compressor.norm."),
    (r"^self_attn\.compressor\.indexer\.position_bias$", "attn.indexer.compressor.ape"),
    (r"^self_attn\.compressor\.kv_proj\.", "attn.compressor.wkv."),
    (r"^self_attn\.compressor\.gate_proj\.", "attn.compressor.wgate."),
    (r"^self_attn\.compressor\.kv_norm\.", "attn.compressor.norm."),
    (r"^self_attn\.compressor\.position_bias$", "attn.compressor.ape"),
    (r"^self_attn\.q_a_proj\.", "attn.wq_a."),
    (r"^self_attn\.q_a_norm\.", "attn.q_norm."),
    (r"^self_attn\.q_b_proj\.", "attn.wq_b."),
    (r"^self_attn\.kv_proj\.", "attn.wkv."),
    (r"^self_attn\.kv_norm\.", "attn.kv_norm."),
    (r"^self_attn\.o_a_proj\.", "attn.wo_a."),
    (r"^self_attn\.o_b_proj\.", "attn.wo_b."),
    (r"^self_attn\.sinks$", "attn.attn_sink"),
    (r"^mlp\.gate\.e_score_correction_bias$", "ffn.gate.bias"),
    (r"^mlp\.gate\.", "ffn.gate."),
    (r"^mlp\.shared_experts\.gate_proj\.", "ffn.shared_experts.w1."),
    (r"^mlp\.shared_experts\.up_proj\.", "ffn.shared_experts.w3."),
    (r"^mlp\.shared_experts\.down_proj\.", "ffn.shared_experts.w2."),
]


def _build_hf_config(layer_types: list[str], mlp_layer_types: list[str]) -> HFDeepseekV4Config:
    """Build a toy HF DeepSeek-V4 config with the given per-layer attention / MoE modes.

    Args:
        layer_types (list[str]): Per-layer attention mode — one of ``"sliding_attention"``,
            ``"compressed_sparse_attention"``, ``"heavily_compressed_attention"``.
        mlp_layer_types (list[str]): Per-layer MoE mode — ``"hash_moe"`` or ``"moe"``.

    Returns:
        HFDeepseekV4Config: Config sized to build in a few MB.
    """
    assert len(layer_types) == len(mlp_layer_types)
    cfg = HFDeepseekV4Config(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_MOE_INTER,
        num_hidden_layers=len(layer_types),
        num_attention_heads=_N_HEADS,
        head_dim=_HEAD_DIM,
        partial_rotary_factor=_QK_ROPE / _HEAD_DIM,
        sliding_window=_SLIDING,
        layer_types=layer_types,
        mlp_layer_types=mlp_layer_types,
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
        n_routed_experts=_N_ROUTED,
        num_experts_per_tok=_N_EXPERTS_PER_TOK,
        n_shared_experts=_N_SHARED,
        num_local_experts=_N_ROUTED,
        o_groups=_O_GROUPS,
        o_lora_rank=_O_LORA,
        q_lora_rank=_Q_LORA,
        index_topk=_INDEX_TOPK,
        index_head_dim=_INDEX_HEAD_DIM,
        index_n_heads=_INDEX_N_HEADS,
        hc_mult=_HC_MULT,
        hc_sinkhorn_iters=_HC_SINKHORN_ITERS,
        hc_eps=_RMS_EPS,
        rms_norm_eps=_RMS_EPS,
        rope_theta=10000.0,
        rope_scaling=dict(_ROPE_SCALING),
        compress_rope_theta=160000.0,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.0,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
        swiglu_limit=10.0,
        max_position_embeddings=_MAX_POS,
        pad_token_id=None,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _hf_to_release_state_dict(hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename a HF DeepSeek-V4 state dict into the released checkpoint's key scheme.

    Args:
        hf_state_dict (dict[str, torch.Tensor]): ``DeepseekV4ForCausalLM.state_dict()``.

    Returns:
        dict[str, torch.Tensor]: Release-format state dict, with the fused routed experts
        split back into per-expert ``w1`` / ``w2`` / ``w3`` tensors.
    """
    out: dict[str, torch.Tensor] = {}
    for key, tensor in hf_state_dict.items():
        if key in _TOP_LEVEL_RENAMES:
            out[_TOP_LEVEL_RENAMES[key]] = tensor
            continue

        match = re.match(r"^model\.layers\.(\d+)\.(.+)$", key)
        assert match is not None, f"unhandled HF key {key!r}; extend the rename tables"
        layer_idx, tail = match.group(1), match.group(2)

        # HF stacks gate and up into one ``[E, 2 * I, H]`` tensor (``F.linear`` then
        # ``chunk(2, -1)``), so the leading ``I`` rows are w1 and the trailing ones w3.
        if tail == "mlp.experts.gate_up_proj":
            for expert_idx in range(tensor.shape[0]):
                gate, up = tensor[expert_idx].split(_MOE_INTER, dim=0)
                out[f"layers.{layer_idx}.ffn.experts.{expert_idx}.w1.weight"] = gate.clone()
                out[f"layers.{layer_idx}.ffn.experts.{expert_idx}.w3.weight"] = up.clone()
            continue
        if tail == "mlp.experts.down_proj":
            for expert_idx in range(tensor.shape[0]):
                out[f"layers.{layer_idx}.ffn.experts.{expert_idx}.w2.weight"] = tensor[expert_idx].clone()
            continue

        for pattern, replacement in _LAYER_TAIL_RENAMES:
            renamed, n_subs = re.subn(pattern, replacement, tail)
            if n_subs:
                out[f"layers.{layer_idx}.{renamed}"] = tensor
                break
        else:
            raise AssertionError(f"unhandled HF layer key {key!r}; extend the rename tables")
    return out


def _release_config_dict(hf_cfg: HFDeepseekV4Config) -> dict:
    """Render a toy HF config as a released-V4-Flash-style ``config.json`` payload.

    The release predates transformers' V4 support and uses the flat legacy schema
    (``compress_ratios``, ``num_hash_layers``) that ``DeepSeekV4Config.from_hf`` reads.

    Args:
        hf_cfg (HFDeepseekV4Config): Source config.

    Returns:
        dict: JSON-serialisable config in the released schema.
    """
    compress_ratios = [_LAYER_TYPE_TO_RATIO[layer_type] for layer_type in hf_cfg.layer_types]
    num_hash_layers = sum(1 for t in hf_cfg.mlp_layer_types if t == "hash_moe")
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "vocab_size": hf_cfg.vocab_size,
        "hidden_size": hf_cfg.hidden_size,
        "moe_intermediate_size": hf_cfg.moe_intermediate_size,
        "num_hidden_layers": hf_cfg.num_hidden_layers,
        "num_attention_heads": hf_cfg.num_attention_heads,
        "num_key_value_heads": 1,
        "head_dim": hf_cfg.head_dim,
        "qk_rope_head_dim": _QK_ROPE,
        "q_lora_rank": hf_cfg.q_lora_rank,
        "o_lora_rank": hf_cfg.o_lora_rank,
        "o_groups": hf_cfg.o_groups,
        "sliding_window": hf_cfg.sliding_window,
        "index_topk": hf_cfg.index_topk,
        "index_head_dim": hf_cfg.index_head_dim,
        "index_n_heads": hf_cfg.index_n_heads,
        "n_routed_experts": hf_cfg.n_routed_experts,
        "n_shared_experts": hf_cfg.n_shared_experts,
        "num_experts_per_tok": hf_cfg.num_experts_per_tok,
        "num_hash_layers": num_hash_layers,
        "compress_ratios": compress_ratios,
        "compress_rope_theta": hf_cfg.compress_rope_theta,
        "rope_theta": hf_cfg.rope_theta,
        # The release spells the rope type as `type`, not `rope_type`.
        "rope_scaling": {"type": "yarn"} | {k: v for k, v in _ROPE_SCALING.items() if k != "rope_type"},
        "max_position_embeddings": hf_cfg.max_position_embeddings,
        "hc_mult": hf_cfg.hc_mult,
        "hc_eps": hf_cfg.hc_eps,
        "hc_sinkhorn_iters": hf_cfg.hc_sinkhorn_iters,
        "rms_norm_eps": hf_cfg.rms_norm_eps,
        "scoring_func": hf_cfg.scoring_func,
        "routed_scaling_factor": hf_cfg.routed_scaling_factor,
        "norm_topk_prob": True,
        "swiglu_limit": hf_cfg.swiglu_limit,
        "hidden_act": hf_cfg.hidden_act,
        "eos_token_id": 1,
        "bos_token_id": 0,
        # MTP is not wired on the XTuner side yet; keep it out of the fixture so the
        # checkpoint has no `mtp.*` keys to account for.
        "num_nextn_predict_layers": 0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }


def _populate_hash_tables(hf_model: DeepseekV4ForCausalLM) -> None:
    """Give every hash-routed layer a non-degenerate ``tid2eid`` table.

    ``DeepseekV4PreTrainedModel._init_weights`` zeroes ``tid2eid`` because the real values
    ship in the checkpoint. Left at zero every token routes to expert 0 ``top_k`` times
    over, which normalises to uniform weights and makes the whole routed-expert path
    degenerate — a hash router that ignores the learned gate scores entirely would still
    match. Fill it with distinct experts per token so routing weights and multi-expert
    dispatch are actually exercised.

    Args:
        hf_model (DeepseekV4ForCausalLM): Model whose hash gates should be populated.
    """
    generator = torch.Generator().manual_seed(0)
    for layer in hf_model.model.layers:
        gate = layer.mlp.gate
        if not hasattr(gate, "tid2eid"):
            continue
        table = torch.stack(
            [torch.randperm(_N_ROUTED, generator=generator)[:_N_EXPERTS_PER_TOK] for _ in range(_VOCAB)]
        )
        gate.tid2eid.copy_(table)


def _export_release_checkpoint(hf_model: DeepseekV4ForCausalLM, out_dir: Path) -> dict[str, torch.Tensor]:
    """Write ``hf_model`` to ``out_dir`` as a DeepSeek-V4-Flash release-format checkpoint.

    Args:
        hf_model (DeepseekV4ForCausalLM): Source model.
        out_dir (Path): Destination directory; created if absent.

    Returns:
        dict[str, torch.Tensor]: The bf16 release-format state dict that was written,
        for use as the expected value in a save round-trip.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dict = _hf_to_release_state_dict(hf_model.state_dict())
    state_dict = {k: v.detach().to(_DTYPE).contiguous() for k, v in state_dict.items()}
    save_file(state_dict, str(out_dir / "model.safetensors"))
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": dict.fromkeys(state_dict, "model.safetensors")})
    )
    (out_dir / "config.json").write_text(json.dumps(_release_config_dict(hf_model.config), indent=2))
    return state_dict


def _hf_model_to_cuda(hf_model: DeepseekV4ForCausalLM) -> DeepseekV4ForCausalLM:
    """Move ``hf_model`` to CUDA and cast its *parameters* — not its buffers — to bf16.

    ``nn.Module.to(dtype=...)`` would also cast the rotary ``inv_freq`` buffers. A bf16
    ``inv_freq`` moves ``cos``/``sin`` by several ULPs once positions grow past ~1, which
    shows up as a whole-percent drift on the compressed-KV rope tail. XTuner loads with
    ``_to_device_dtype(..., skip_buffers_dtype=True)`` and keeps every buffer at its
    checkpoint dtype, so the reference has to do the same or the two sides rotate against
    different bases.

    Args:
        hf_model (DeepseekV4ForCausalLM): Model on CPU.

    Returns:
        DeepseekV4ForCausalLM: The same model, on CUDA, in eval mode.
    """
    hf_model = hf_model.to(device="cuda")
    for param in hf_model.parameters():
        param.data = param.data.to(_DTYPE)
    return hf_model.eval()


def _read_safetensors_dir(path: Path) -> dict[str, torch.Tensor]:
    """Load every tensor from every ``*.safetensors`` shard under ``path``.

    Args:
        path (Path): Directory holding one or more safetensors shards.

    Returns:
        dict[str, torch.Tensor]: Merged key → tensor mapping.
    """
    from safetensors import safe_open

    merged: dict[str, torch.Tensor] = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            for key in handle.keys():
                merged[key] = handle.get_tensor(key)
    return merged


def _build_xtuner_model(ckpt_dir: Path, device: str) -> DeepSeekV4:
    """Build an XTuner DeepSeek-V4 from a release-format checkpoint via the HF path.

    Args:
        ckpt_dir (Path): Directory produced by :func:`_export_release_checkpoint`.
        device (str): Target device, e.g. ``"cuda"``.

    Returns:
        DeepSeekV4: Model with the checkpoint's weights loaded.
    """
    cfg = get_model_config_from_hf(ckpt_dir)
    assert isinstance(cfg, DeepSeekV4Config)
    cfg.dispatcher = None  # eager experts, no all2all
    cfg.compile_cfg = False
    cfg.ep_size = 1
    cfg.attention.backend = "native"  # avoid FlashMLA / cudnn in unit tests
    cfg.attention.indexer_backend = "native"
    # HF's routers run the gate in the input dtype (bf16); XTuner defaults to an fp32
    # gate for routing stability. Pin to HF's dtype so parity is measurable — the fp32
    # path remains the production default.
    cfg.router_compute_dtype = "native"
    with torch.device("meta"):
        model = cfg.build()
    model = model._to_device_dtype(dtype=_DTYPE, skip_buffers_dtype=True)
    model.from_hf(ckpt_dir)
    return model.to(device).eval()


def _xtuner_logits(model: DeepSeekV4, input_ids: torch.Tensor) -> torch.Tensor:
    """Run one packed-varlen forward and return the logits.

    Args:
        model (DeepSeekV4): XTuner model.
        input_ids (torch.Tensor): ``[1, S]`` token ids.

    Returns:
        torch.Tensor: Logits as returned by the model forward.
    """
    seq_ctx = SequenceContext.from_input_ids(input_ids=(input_ids,))
    loss_cfg = CELossConfig(mode="eager")
    loss_ctx = loss_cfg.build(data={"shifted_labels": input_ids.clone()}, sp_mesh=None)
    loss_ctx = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])[0]
    with torch.no_grad():
        output = model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
    return output["logits"]


# ─── Tests ──────────────────────────────────────────────────────────────────────


class TestDeepSeekV4Checkpoint:
    """Config parsing and HF checkpoint load / save behaviour. CPU only."""

    def test_release_checkpoint_round_trip(self, tmp_path: Path) -> None:
        """``from_hf`` then ``save_hf`` reproduces the source checkpoint exactly.

        A single CSA layer already exercises every key family the mapping has to handle:
        both HC triples, the attention LoRA chain, the compressor, the nested Indexer,
        the routed + shared experts, and the model-level ``hc_head`` / ``embed`` / ``head``.
        """
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29591")
            dist.init_process_group("gloo", rank=0, world_size=1)

        torch.manual_seed(0)
        hf_cfg = _build_hf_config(["compressed_sparse_attention"], ["moe"])
        hf_model = DeepseekV4ForCausalLM(hf_cfg)
        src_dir = tmp_path / "release"
        expected = _export_release_checkpoint(hf_model, src_dir)

        cfg = get_model_config_from_hf(src_dir)
        cfg.dispatcher = None
        cfg.compile_cfg = False
        with torch.device("meta"):
            model = cfg.build()
        model = model._to_device_dtype(dtype=_DTYPE, skip_buffers_dtype=True)
        loaded_keys, unloaded_keys, missing_keys = model.from_hf(src_dir)

        # Every checkpoint tensor must be claimed by a parameter and vice versa. This is
        # the assertion the old ">= 90% key coverage" heuristic was standing in for.
        assert not unloaded_keys, f"checkpoint keys never consumed: {sorted(unloaded_keys)}"
        assert not missing_keys, f"parameters with no checkpoint key: {sorted(missing_keys)}"
        assert loaded_keys, "from_hf loaded nothing"

        dst_dir = tmp_path / "resaved"
        model.save_hf(dst_dir)
        resaved = _read_safetensors_dir(dst_dir)

        assert set(resaved) == set(expected), (
            f"save_hf key set differs; "
            f"only in source: {sorted(set(expected) - set(resaved))}, "
            f"only in resaved: {sorted(set(resaved) - set(expected))}"
        )
        mismatched = [key for key, value in expected.items() if not torch.equal(resaved[key], value)]
        assert not mismatched, f"save_hf changed {len(mismatched)} tensor(s) bitwise: {mismatched[:8]}"

    def test_config_from_released_checkpoint(self) -> None:
        """``DeepSeekV4Config.from_hf`` round-trips the real released ``config.json``."""
        path = Path(os.environ.get(_REAL_BF16_PATH_ENV, _REAL_BF16_DEFAULT))
        if not (path / "config.json").exists():
            pytest.skip(f"{_REAL_BF16_PATH_ENV} not set / path missing; cannot test from_hf")

        cfg = DeepSeekV4Config.from_hf(path)
        assert cfg.num_hidden_layers == 43
        assert cfg.num_hash_layers == 3
        assert cfg.hidden_size == 4096
        assert cfg.n_routed_experts == 256
        assert cfg.n_shared_experts == 1
        assert cfg.num_experts_per_tok == 6
        assert cfg.vocab_size == 129280
        assert cfg.router.scoring_func == "sqrtsoftplus"
        assert cfg.attention.head_dim == 512
        assert cfg.attention.q_lora_rank == 1024
        assert cfg.attention.o_lora_rank == 1024
        assert cfg.attention.o_groups == 8
        assert cfg.attention.qk_rope_head_dim == 64
        assert cfg.attention.num_key_value_heads == 1
        assert cfg.hc_cfg.hc_mult == 4
        assert cfg.hc_cfg.hc_sinkhorn_iters == 20
        assert cfg.swiglu_limit == 10.0
        assert cfg.mtp_config is not None and cfg.mtp_config.num_layers == 1

        # Per the release, layers 0/1 are pure sliding-window and 2..42 alternate 4/128.
        ratios = cfg.rope_parameters_cfg.compress_ratios
        assert ratios is not None
        assert len(ratios) == cfg.num_hidden_layers
        assert ratios[:4] == [0, 0, 4, 128]
        assert cfg.rope_parameters_cfg.compress_rope_theta == 160000.0

    def test_hash_layer_aux_loss_gated_off(self, tmp_path: Path) -> None:
        """Hash-routed layers skip aux loss; score-routed layers keep it."""
        torch.manual_seed(0)
        hf_cfg = _build_hf_config(
            ["sliding_attention"] * 4,
            ["hash_moe"] * 3 + ["moe"],
        )
        src_dir = tmp_path / "release"
        _export_release_checkpoint(DeepseekV4ForCausalLM(hf_cfg), src_dir)

        cfg = get_model_config_from_hf(src_dir)
        cfg.dispatcher = None
        cfg.compile_cfg = False
        with torch.device("meta"):
            model = cfg.build()
        assert isinstance(model, DeepSeekV4)
        assert cfg.num_hash_layers == 3

        for idx in range(cfg.num_hash_layers):
            assert model._should_compute_aux_loss(idx) is False, f"layer {idx} (hash-routed) should skip aux loss"
        for idx in range(cfg.num_hash_layers, cfg.num_hidden_layers):
            assert model._should_compute_aux_loss(idx) is True, f"layer {idx} (score-routed) should compute aux loss"


@pytest.mark.gpu
class TestDeepSeekV4Parity:
    """Forward parity against the HF reference, per layer flavour and combined."""

    @pytest.fixture(autouse=True)
    def _seed(self) -> None:
        torch.manual_seed(0)

    @pytest.fixture(autouse=True)
    def _cutlass_group_gemm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The default ``m_grouped_gemm_TMA_triton3_4`` expert kernel fails to compile under
        # Triton 3.5.1 (``PassManager::run failed`` in the ``ttng.tensormap_create``
        # pipeliner), which takes down ``MoEBlock.forward``. ``XTUNER_USE_CUTLASS_GROUP_GEMM``
        # selects the cutlass backend instead, but it is read once at ``xtuner.v1.ops.moe``
        # import time and the root conftest already imported xtuner by then — so rebind the
        # symbol the grouped linear actually calls. Drop this once the Triton kernel builds.
        from xtuner.v1.module.grouped_linear import moe_group_linear
        from xtuner.v1.ops.moe.cuda import cutlass_group_gemm

        monkeypatch.setattr(moe_group_linear, "group_gemm", cutlass_group_gemm)

    @pytest.fixture(autouse=True)
    def _hf_parity(self) -> Iterator[None]:
        # ``XTUNER_V4_HF_PARITY`` puts every V4 site that deliberately diverges from HF —
        # the HC bf16 fast paths, the fp32 gathered attention core, the reference-faithful
        # low-precision Indexer score — onto HF's exact op sequence, which is what makes a
        # whole-model forward reproduce HF bitwise. See ``xtuner.v1.utils.hf_parity``.
        previous = set_hf_parity(True)
        try:
            yield
        finally:
            set_hf_parity(previous)

    def _build_pair(
        self,
        tmp_path: Path,
        layer_types: list[str],
        mlp_layer_types: list[str],
    ) -> tuple[DeepseekV4ForCausalLM, DeepSeekV4]:
        """Build matched HF / XTuner models from one exported release checkpoint."""
        hf_cfg = _build_hf_config(layer_types, mlp_layer_types)
        hf_model = DeepseekV4ForCausalLM(hf_cfg)
        _populate_hash_tables(hf_model)
        src_dir = tmp_path / "release"
        _export_release_checkpoint(hf_model, src_dir)
        return _hf_model_to_cuda(hf_model), _build_xtuner_model(src_dir, device="cuda")

    def _assert_bitwise(
        self,
        tmp_path: Path,
        layer_types: list[str],
        mlp_layer_types: list[str],
        *,
        seq_len: int,
        seeds: int = 3,
    ) -> None:
        """Assert the whole-model logits are ``torch.equal`` to HF's, on every seed.

        No tolerance: under HF parity there is nothing left for one to absorb, and a
        tolerance wide enough to pass would also have passed every structural bug this
        file has caught (wrong rope base on compressed layers, one-sided Hadamard in the
        Indexer, a hash router that ignored the gate, ``hc_head`` skipping the parity
        path). ``torch.equal`` cannot be satisfied by accident.

        ``lm_head`` computes in bf16 and upcasts, so HF's bf16 logits are compared as
        ``.float()`` — a lossless widening of the same bits, not a relaxation.
        """
        for seed in range(seeds):
            torch.manual_seed(seed)
            hf_model, xtuner_model = self._build_pair(tmp_path / f"seed{seed}", layer_types, mlp_layer_types)

            input_ids = torch.randint(0, _VOCAB, (1, seq_len), device="cuda", dtype=torch.long)
            with torch.no_grad():
                hf_logits = hf_model(input_ids=input_ids).logits
            xt_logits = _xtuner_logits(xtuner_model, input_ids).view_as(hf_logits)

            expected = hf_logits.float()
            if not torch.equal(xt_logits, expected):
                diff = (xt_logits - expected).abs()
                identical = (xt_logits == expected).float().mean().item()
                raise AssertionError(
                    f"logits not bitwise equal to HF (seed {seed}): "
                    f"max|Δ|={diff.max().item():.3e}, {identical:.2%} of elements identical"
                )

    def test_sliding_layer_bitwise(self, tmp_path: Path) -> None:
        """Hash-routed MoE on a sliding-window layer: no compressor, no Indexer."""
        self._assert_bitwise(tmp_path, ["sliding_attention"], ["hash_moe"], seq_len=32)

    def test_sliding_layer_bitwise_long(self, tmp_path: Path) -> None:
        """Same layer past the window, so most queries see a strict sub-range of the KV.

        At ``seq_len <= sliding_window`` every query attends to a prefix and the gathered
        and dense reductions happen to coincide; only beyond the window does the parity
        path's dense contraction actually earn its keep.
        """
        self._assert_bitwise(tmp_path, ["sliding_attention"], ["moe"], seq_len=256)

    def test_csa_layer_bitwise(self, tmp_path: Path) -> None:
        """compress_ratio=4: KVCompressor plus Indexer top-K over the sliding window."""
        self._assert_bitwise(tmp_path, ["compressed_sparse_attention"], ["moe"], seq_len=64)

    def test_csa_layer_bitwise_long(self, tmp_path: Path) -> None:
        """CSA where the Indexer must actually choose: 64 compressed entries, top-K of 8."""
        self._assert_bitwise(tmp_path, ["compressed_sparse_attention"], ["moe"], seq_len=256)

    def test_hca_layer_bitwise(self, tmp_path: Path) -> None:
        """compress_ratio=128: KVCompressor with a deterministic positional gather.

        Needs a long enough sequence for the compressed chunks to be non-trivial — at
        ratio 128 the last query of a 256-token sequence sees two compressed chunks.
        """
        self._assert_bitwise(tmp_path, ["heavily_compressed_attention"], ["moe"], seq_len=256)

    def test_minimal_v4_flash_bitwise(self, tmp_path: Path) -> None:
        """A minimal V4-Flash carrying every layer flavour at once.

        Mirrors the release's layout: hash-routed sliding layers first, then the
        score-routed CSA / HCA alternation. Stacking is what makes this strictly stronger
        than the single-layer cases — each layer feeds the next, so any per-layer
        divergence compounds instead of staying local.
        """
        self._assert_bitwise(
            tmp_path,
            ["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"],
            ["hash_moe", "moe", "moe"],
            seq_len=256,
        )

    def test_stacked_v4_flash_bitwise(self, tmp_path: Path) -> None:
        """Two full CSA/HCA cycles, so every layer flavour is exercised twice in sequence."""
        self._assert_bitwise(
            tmp_path,
            ["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"] * 2,
            ["hash_moe"] + ["moe"] * 5,
            seq_len=384,
            seeds=2,
        )

    def test_csa_indexer_topk_matches_hf(self, tmp_path: Path) -> None:
        """The Indexer selects exactly HF's compressed entries, index set for index set.

        Pins the selection itself rather than its blurred effect on the logits — a logits
        tolerance wide enough to survive a stacked model would not notice the Indexer
        scoring against un-rotated keys, which is what it did before
        ``rotate_activation`` was applied to the compressed stream as well as to ``q``.
        One layer on a bitwise-identical input, and a sequence short enough
        (64 tokens → 16 compressed entries for an ``index_topk`` of 8) that no two scores
        tie within bf16, so the comparison is exact rather than probabilistic.
        """
        hf_model, xtuner_model = self._build_pair(tmp_path, ["compressed_sparse_attention"], ["moe"])

        captured: dict[str, torch.Tensor] = {}
        hf_model.model.layers[0].self_attn.compressor.indexer.register_forward_hook(
            lambda module, args, output: captured.__setitem__("hf", output)
        )
        xtuner_model.layers["0"].self_attn.indexer.register_forward_hook(
            lambda module, args, output: captured.__setitem__("xt", output)
        )

        input_ids = torch.randint(0, _VOCAB, (1, 64), device="cuda", dtype=torch.long)
        with torch.no_grad():
            hf_model(input_ids=input_ids)
        _xtuner_logits(xtuner_model, input_ids)

        # ``-1`` marks a slot the causal horizon rejected; both sides emit it, but the
        # padding is not order-aligned, so compare the selected sets per query.
        hf_topk = captured["hf"].reshape(64, -1).tolist()
        xt_topk = captured["xt"].reshape(64, -1).tolist()
        mismatched = [
            query
            for query, (hf_row, xt_row) in enumerate(zip(hf_topk, xt_topk))
            if sorted(e for e in hf_row if e >= 0) != sorted(e for e in xt_row if e >= 0)
        ]
        assert not mismatched, f"Indexer picked different compressed entries for queries {mismatched}"
