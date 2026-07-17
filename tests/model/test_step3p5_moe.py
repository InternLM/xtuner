import itertools
import math
import os
import re
import unittest

import parametrize
import torch
import torch.distributed as dist
import torch.nn as nn

from xtuner.v1.data_proto import SequenceContext
from xtuner._testing import DeterministicDDPTestCase


# Step-3.5-Flash (MoE, hybrid full/sliding attention, head-wise gate, per-layer RoPE).
STEP3P5_PATH = os.environ.get("STEP3P5_PATH")


def _patch_hf_step3p5(hf_path, hf_cfg, llama3):
    """Make the trust_remote_code modeling importable/runnable on the installed transformers.

    The shipped ``modeling_step3p5.py`` targets a different transformers version: its
    ``Step3p5RotaryEmbedding`` crashes here (``ROPE_INIT_FUNCTIONS['default']`` missing; llama3 reads a
    list ``rope_theta``). We replace only the rotary with a version-independent implementation that uses
    the canonical default / llama3 inverse-frequency formulas — the same math XTuner uses and that the
    rotary-parity test pins to the reference. Everything else (attention gate, qk_norm, MoE router,
    experts, SwiGLU clamp, zero-centered RMSNorm) stays genuine HF.
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    mod = __import__(
        get_class_from_dynamic_module("modeling_step3p5.Step3p5Attention", hf_path).__module__,
        fromlist=["x"],
    )
    # Capture per-layer rope constants once: HF mutates the config during layer construction.
    theta = list(hf_cfg.rope_theta)
    partial = list(hf_cfg.partial_rotary_factors)
    yarn = list(hf_cfg.yarn_only_types)
    layer_types = list(hf_cfg.layer_types)
    head_dim = hf_cfg.head_dim

    def _llama3(inv):
        old = llama3["original_max_position_embeddings"]
        lo, hi, f = llama3["low_freq_factor"], llama3["high_freq_factor"], llama3["factor"]
        low_wl, high_wl = old / lo, old / hi
        wl = 2 * math.pi / inv
        iv = torch.where(wl > low_wl, inv / f, inv)
        sm = (old / wl - lo) / (hi - lo)
        sa = (1 - sm) * iv / f + sm * iv
        return torch.where(~(wl < high_wl) * ~(wl > low_wl), sa, iv)

    class _FixedRotary(nn.Module):
        def __init__(self, config, device=None, layer_idx=None):
            super().__init__()
            dim = int(head_dim * partial[layer_idx])
            inv = 1.0 / (theta[layer_idx] ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
            if layer_types[layer_idx] in yarn:
                inv = _llama3(inv)
            self.attention_scaling = 1.0
            self.register_buffer("inv_freq", inv, persistent=False)

        @torch.no_grad()
        def forward(self, x, position_ids):
            inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            pos = position_ids[:, None, :].float()
            with torch.autocast(device_type=x.device.type, enabled=False):
                freqs = (inv.float().to(x.device) @ pos.float().to(x.device)).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    mod.Step3p5RotaryEmbedding = _FixedRotary
    return mod, _FixedRotary


@unittest.skipIf(STEP3P5_PATH is None, "STEP3P5_PATH env var is required for Step-3.5 parity tests")
class TestStep3p5MoE(DeterministicDDPTestCase):
    def _build(self):
        from transformers import AutoConfig

        from xtuner.v1.model import get_model_config_from_hf

        cfg = get_model_config_from_hf(STEP3P5_PATH)
        cfg.compile_cfg = False
        hf_cfg = AutoConfig.from_pretrained(STEP3P5_PATH, trust_remote_code=True)
        hf_cfg._attn_implementation = "eager"
        llama3 = {
            "factor": cfg.rope_factor,
            "low_freq_factor": cfg.rope_low_freq_factor,
            "high_freq_factor": cfg.rope_high_freq_factor,
            "original_max_position_embeddings": cfg.rope_original_max_position_embeddings,
        }
        return cfg, hf_cfg, llama3

    @parametrize.parametrize("device", [("cuda",)])
    def test_rotary_inv_freq_parity(self, device):
        # Per-profile inverse frequencies must match the canonical default / llama3 formulas bitwise.
        self.create_pg(device)
        cfg, _, _ = self._build()

        full = cfg.build_layer_rotary("full_attention").inv_freq
        dim = int(cfg.attention.head_dim * cfg.full_partial_rotary_factor)
        base = 1.0 / (cfg.full_rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        from xtuner.v1.model.moe.step3p5 import Step3p5RotaryEmbedding

        ref_full = Step3p5RotaryEmbedding._apply_llama3_smoothing(
            base,
            factor=cfg.rope_factor,
            low_freq_factor=cfg.rope_low_freq_factor,
            high_freq_factor=cfg.rope_high_freq_factor,
            original_max_position_embeddings=cfg.rope_original_max_position_embeddings,
        )
        self.assertTrue(torch.equal(full, ref_full), "full-attention inv_freq not bitwise vs llama3 reference")

        sliding = cfg.build_layer_rotary("sliding_attention").inv_freq
        sdim = int(cfg.sliding_attention.head_dim * cfg.sliding_partial_rotary_factor)
        ref_sliding = 1.0 / (cfg.sliding_rope_theta ** (torch.arange(0, sdim, 2, dtype=torch.int64).float() / sdim))
        self.assertTrue(torch.equal(sliding, ref_sliding), "sliding-attention inv_freq not bitwise vs default reference")
        dist.barrier()

    @parametrize.parametrize("device,layer_idx", [("cuda", 4), ("cuda", 3), ("cuda", 44)])
    def test_decoder_attention_bitwise_parity(self, device, layer_idx):
        # Attention sub-block (q/k/v + qk_norm + per-layer RoPE + partial rotary + head-wise gate +
        # sliding window) must match HF bitwise under XTUNER_HF_IMPL (eager). Covers a full layer (4),
        # a sliding layer (3), and a full layer with SwiGLU clamp (44). Memory-light: only the attention
        # of one layer is materialized on each side.
        from xtuner.v1.utils import HFCheckpointLoader

        self.create_pg(device)
        with self.hf_impl():
            cfg, hf_cfg, llama3 = self._build()
            loader = HFCheckpointLoader(STEP3P5_PATH)
            mod, FixedRotary = _patch_hf_step3p5(STEP3P5_PATH, hf_cfg, llama3)
            layer_type = cfg.layers_type[layer_idx]

            with torch.device("meta"):
                model = cfg.build()
            xt_layer = model.layers[str(layer_idx)]
            xt_attn = xt_layer.self_attn
            xt_attn.to_empty(device="cuda")
            xt_attn.to(torch.bfloat16)
            for n, p in xt_attn.named_parameters():
                key = model.to_hf_key_list(f"layers.{layer_idx}.self_attn.{n}")[0]
                p.data.copy_(loader.load(key).to(p.device, p.dtype))
            xt_layer.rotary_emb.inv_freq = cfg.build_layer_rotary(layer_type).inv_freq.to("cuda")

            hf_attn = mod.Step3p5Attention(hf_cfg, layer_idx).to("cuda").to(torch.bfloat16).eval()
            for n, p in hf_attn.named_parameters():
                p.data.copy_(loader.load(f"model.layers.{layer_idx}.self_attn.{n}").to(p.device, p.dtype))
            # `.to(bf16)` rounds the fp32 inv_freq buffer; restore the fp32 value (kept fp32 in production).
            hf_attn.rotary_emb.inv_freq = FixedRotary(hf_cfg, layer_idx=layer_idx).inv_freq.to("cuda")

            seq = 16
            torch.manual_seed(0)
            ids = torch.randint(0, 1000, (1, seq), device="cuda")
            seq_ctx = SequenceContext.from_input_ids(input_ids=(ids,))
            seq_ctx.to("cuda")
            x = torch.randn(1, seq, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)

            o_xt = xt_attn(x, xt_layer.rotary_emb(x, seq_ctx.position_ids), seq_ctx)["projected_output"]
            mask = torch.triu(
                torch.full((seq, seq), float("-inf"), device="cuda", dtype=torch.bfloat16), diagonal=1
            )[None, None]
            o_hf, _ = hf_attn(x, attention_mask=mask, position_ids=seq_ctx.position_ids)

        diff = (o_hf.float() - o_xt.float().reshape(o_hf.shape)).abs().max().item()
        self.assertEqual(diff, 0.0, f"layer {layer_idx} [{layer_type}] attention not bitwise: max diff {diff}")
        dist.barrier()

    @parametrize.parametrize("device,layer_idx", [("cuda", 4), ("cuda", 44)])
    def test_decoder_layer_tolerance_parity(self, device, layer_idx):
        # Full MoE decoder layer (attention + router + fused experts + shared expert + SwiGLU clamp).
        # Not bitwise: XTuner uses a bf16 grouped GEMM while the HF reference upcasts each expert matmul
        # to fp32 (`MoELinear.float()`); the established MoE bar is tolerance (see test_qwen3_moe). The
        # router top-k *indices* must still match exactly (both compute sigmoid+bias+top-k in fp32).
        from xtuner.v1.utils import HFCheckpointLoader

        self.create_pg(device)
        with self.hf_impl():
            cfg, hf_cfg, llama3 = self._build()
            loader = HFCheckpointLoader(STEP3P5_PATH)
            mod, FixedRotary = _patch_hf_step3p5(STEP3P5_PATH, hf_cfg, llama3)
            layer_type = cfg.layers_type[layer_idx]

            with torch.device("meta"):
                model = cfg.build()
            xt_layer = model.layers[str(layer_idx)]
            xt_layer.to_empty(device="cuda")
            for p in xt_layer.parameters():
                p.data = p.data.to(torch.bfloat16)  # params bf16; the fp32 router-bias buffer is untouched
            xt_layer.rotary_emb.inv_freq = cfg.build_layer_rotary(layer_type).inv_freq.to("cuda")
            self._load_xt_layer(model, xt_layer, layer_idx, loader)

            hf_layer = mod.Step3p5DecoderLayer(hf_cfg, layer_idx).to("cuda").eval()
            for p in hf_layer.parameters():
                p.data = p.data.to(torch.bfloat16)
            for n, p in hf_layer.named_parameters():
                if "router_bias" in n:
                    p.data = p.data.to(torch.float32)  # need_fp32_gate
            n_exp = cfg.n_routed_experts

            def _hf_tensor(name):
                # The HF reference layer expects fused 3-D expert tensors, but the checkpoint stores
                # experts split per expert; restack them for the HF side.
                m = re.match(r"moe\.(gate_proj|up_proj|down_proj)\.weight$", name)
                if m is not None:
                    proj = m.group(1)
                    return torch.stack(
                        [loader.load(f"model.layers.{layer_idx}.moe.experts.{i}.{proj}.weight") for i in range(n_exp)]
                    )
                return loader.load(f"model.layers.{layer_idx}.{name}")

            for n, p in itertools.chain(
                hf_layer.named_parameters(),
                ((n, b) for n, b in hf_layer.named_buffers() if not n.endswith("inv_freq")),
            ):
                p.data.copy_(_hf_tensor(n).to(p.device, p.dtype))
            hf_layer.self_attn.rotary_emb.inv_freq = FixedRotary(hf_cfg, layer_idx=layer_idx).inv_freq.to("cuda")

            seq = 64
            torch.manual_seed(0)
            ids = torch.randint(0, 1000, (1, seq), device="cuda")
            seq_ctx = SequenceContext.from_input_ids(input_ids=(ids,))
            seq_ctx.to("cuda")
            x = torch.randn(1, seq, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)

            o_xt = xt_layer(x, seq_ctx=seq_ctx, position_embeddings=None)
            o_xt = o_xt[0] if isinstance(o_xt, tuple) else o_xt
            mask = torch.triu(
                torch.full((seq, seq), float("-inf"), device="cuda", dtype=torch.bfloat16), diagonal=1
            )[None, None]
            o_hf = hf_layer(x, attention_mask=mask, position_ids=seq_ctx.position_ids)
            o_hf = o_hf[0] if isinstance(o_hf, tuple) else o_hf

            # Router top-k indices: exact match.
            xt_topk = xt_layer.gate(x)["topk_ids"].sort(dim=-1).values
            hf_logits = (x.view(-1, cfg.hidden_size).float() @ hf_layer.moe.gate.weight.t().float())
            hf_prob = hf_logits.sigmoid()
            _, hf_idx = torch.topk(hf_prob + hf_layer.moe.router_bias.unsqueeze(0), k=cfg.num_experts_per_tok, dim=1)
            hf_topk = hf_idx.sort(dim=-1).values

        rel = (o_hf.float() - o_xt.float().reshape(o_hf.shape)).abs().max().item() / o_hf.float().abs().max().item()
        self.assertTrue(torch.equal(xt_topk, hf_topk), f"layer {layer_idx} router top-k indices differ")
        self.assertLess(rel, 0.02, f"layer {layer_idx} [{layer_type}] full-layer rel error too high: {rel}")
        dist.barrier()

    @staticmethod
    def _load_xt_layer(model, layer, layer_idx, loader):
        # Load one XTuner decoder layer's params (and persistent router-bias buffer). The fused expert
        # grouped-linear maps to many per-expert HF keys whose interleaved order matches the fused
        # layout, so the default `safetensors_to_params` concatenates them along dim 0.
        named = itertools.chain(
            layer.named_parameters(),
            ((n, b) for n, b in layer.named_buffers() if not n.endswith("inv_freq")),
        )
        for name, tensor in named:
            full = f"layers.{layer_idx}.{name}"
            hf_keys = model.to_hf_key_list(full)
            tensors = [loader.load(k) for k in hf_keys]
            assert all(t is not None for t in tensors), (full, hf_keys)
            # dim=0: per-expert keys concatenate along the fused grouped-linear's expert-major dim.
            model.safetensors_to_params(tensors, tensor.data, full, None, None, 0)

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "1"))
