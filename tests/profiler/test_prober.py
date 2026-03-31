"""Tests for profiler/prober_utils.py and AccProber.

Two levels of tests:
1. TestProberRegistration  - structural unit tests, no dist required.
   Verify that register_prober_list() wraps GatedDeltaNet (and keeps wrapping
   MultiHeadAttention / MultiLatentAttention) by inspecting the replaced
   forward binding, without running an actual forward pass.

2. TestAccProberForwardRecords - functional integration tests, uses
   DeterministicDDPTestCase (single-GPU 1-process group) so that
   dist.get_rank() is available when AccProber dumps records.
   Builds a tiny Qwen3.5-style MoE model (4 layers: 3 GatedDeltaNet +
   1 MHA), runs a forward pass, and asserts that the JSONL file contains
   attention tensors for both attention types.
"""

import json
import re
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.module.attention import GatedDeltaNet, GatedDeltaNetConfig, MHAConfig
from xtuner.v1.module.rope import RopeScalingConfig
from xtuner.v1.module.attention.mha import MultiHeadAttention
from xtuner.v1.profiler.prober import AccProber, ProberList
from xtuner.v1.profiler.prober_utils import register_prober_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_small_model_config() -> Qwen3_5_VLTextMoE35BA3BConfig:
    """Return a tiny Qwen3.5 config with 4 layers.

    With the Qwen3.5 layers_type formula ``(i+1) % 4``:
      layer 0 → linear_attention (GatedDeltaNet)
      layer 1 → linear_attention (GatedDeltaNet)
      layer 2 → linear_attention (GatedDeltaNet)
      layer 3 → full_attention   (MHA)
    """
    return Qwen3_5_VLTextMoE35BA3BConfig(
        num_hidden_layers=4,
        rope_scaling_cfg=RopeScalingConfig(type="default"),  # standard RotaryEmbedding, no mrope
        balancing_loss_cfg=None,
        z_loss_cfg=None,
        compile_cfg=False,
    )


def _reset_acc_prober():
    """Reset AccProber class state between tests."""
    AccProber.dump_dir = None
    AccProber.profile_step = None
    AccProber.initialized = False
    AccProber.cur_step = 0
    AccProber.cur_micro_batch_iter = 0
    AccProber._skip_flag = True
    AccProber.forward_records = []
    AccProber._pending_tensors = []
    ProberList.prober_list = []


# ---------------------------------------------------------------------------
# Functional integration tests (dist required for AccProber dump)
# ---------------------------------------------------------------------------

class TestAccProberForwardRecords(DeterministicDDPTestCase):
    """Run a forward pass through a tiny Qwen3.5 model and check the JSONL."""

    def test_acc_prober_records_both_attention_types(self):
        self.create_pg("cuda")
        _reset_acc_prober()

        config = _make_small_model_config()
        # Build directly on CPU then move to GPU; no meta-device / FSDP needed here.
        # init_weights() is intentionally skipped: it doesn't handle GatedDeltaNet's
        # custom parameters (dt_bias, A_log), and their __init__ values are fine for
        # a forward-pass smoke test.
        model = config.build().to(torch.bfloat16).cuda()

        PROBE_STEP = 1
        SEQ_LEN = 16

        with tempfile.TemporaryDirectory() as tmp_dir:
            ProberList.setup(Path(tmp_dir), [PROBE_STEP], ["AccProber"])
            register_prober_list(model)

            ProberList.set_step(PROBE_STEP)
            ProberList.set_micro_batch_iter(0)

            input_ids = torch.randint(0, config.vocab_size, (1, SEQ_LEN), device="cuda")
            shifted_labels = input_ids[:, 1:].clone()
            shift_input_ids = input_ids[:, :-1]

            seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids,))

            loss_cfg = CELossConfig()
            LossCtx = loss_cfg.loss_ctx_cls
            loss_ctx = loss_cfg.build(shifted_labels=shifted_labels, sp_mesh=None)
            loss_ctx = LossCtx.build_batches([loss_ctx])[0]

            with torch.no_grad():
                model(seq_ctx=seq_ctx, loss_ctx=loss_ctx)

            # Dump records to disk (requires dist.get_rank())
            AccProber.after_micro_iter_forward()

            dump_file = (
                Path(tmp_dir)
                / "acc_prober"
                / f"Step_{PROBE_STEP}_MicroIter_0_RANK_{dist.get_rank()}_forward_records.jsonl"
            )
            self.assertTrue(dump_file.exists(), f"Dump file not found: {dump_file}")

            records = [json.loads(line) for line in dump_file.read_text().splitlines() if line.strip()]
            names = [r["name"] for r in records]

            # ---- Attention tensors must appear ----
            # Layer 3 is MHA ("self_attn"), layers 0-2 are GatedDeltaNet (also "self_attn").
            # All attention modules are registered under their named_modules() path,
            # which is always "layers.N.self_attn" regardless of attention type.
            attn_before = [n for n in names if "self_attn" in n and "[before]hidden_states" in n]
            attn_after  = [n for n in names if "self_attn" in n and "[after]outputs" in n]

            self.assertGreater(len(attn_before), 0, "Expected at least one [before]hidden_states for self_attn")
            self.assertGreater(len(attn_after),  0, "Expected at least one [after]outputs for self_attn")

            # Specifically, layers 0-2 (GatedDeltaNet) and layer 3 (MHA) must each appear.
            linear_attn_before = [n for n in attn_before if any(f"layers.{i}.self_attn" in n for i in range(3))]
            full_attn_before   = [n for n in attn_before if "layers.3.self_attn" in n]

            self.assertGreater(
                len(linear_attn_before), 0,
                "Expected GatedDeltaNet (layers 0-2) attention tensors in records",
            )
            self.assertGreater(
                len(full_attn_before), 0,
                "Expected MHA (layer 3) attention tensors in records",
            )

        _reset_acc_prober()


# ---------------------------------------------------------------------------
# Compiled-mode integration test
# ---------------------------------------------------------------------------

class TestAccProberForwardRecordsCompiled(DeterministicDDPTestCase):
    """Same as TestAccProberForwardRecords but with torch.compile enabled.

    Verifies that the buffer-based record_tensor approach captures the same
    module-level tensors as non-compiled mode (decoder-layer before/after,
    self_attn before/after, experts before/after) without using
    @torch._dynamo.disable on any wrapper.
    """

    def test_acc_prober_records_with_compile(self):
        self.create_pg("cuda")
        _reset_acc_prober()
        torch._dynamo.reset()

        config = _make_small_model_config()
        # compile_cfg=True enables torch.compile via default_compile_cfg
        config.compile_cfg = True
        model = config.build().to(torch.bfloat16).cuda()

        PROBE_STEP = 1
        SEQ_LEN = 16

        with tempfile.TemporaryDirectory() as tmp_dir:
            ProberList.setup(Path(tmp_dir), [PROBE_STEP], ["AccProber"])
            register_prober_list(model)

            ProberList.set_step(PROBE_STEP)
            ProberList.set_micro_batch_iter(0)

            input_ids = torch.randint(0, config.vocab_size, (1, SEQ_LEN), device="cuda")
            shifted_labels = input_ids[:, 1:].clone()
            shift_input_ids = input_ids[:, :-1]

            seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids,))
            loss_cfg = CELossConfig()
            LossCtx = loss_cfg.loss_ctx_cls
            loss_ctx = loss_cfg.build(shifted_labels=shifted_labels, sp_mesh=None)
            loss_ctx = LossCtx.build_batches([loss_ctx])[0]

            with torch.no_grad():
                model(seq_ctx=seq_ctx, loss_ctx=loss_ctx)

            AccProber.after_micro_iter_forward()

            dump_file = (
                Path(tmp_dir)
                / "acc_prober"
                / f"Step_{PROBE_STEP}_MicroIter_0_RANK_{dist.get_rank()}_forward_records.jsonl"
            )
            self.assertTrue(dump_file.exists(), f"Compiled dump file not found: {dump_file}")

            records = [json.loads(line) for line in dump_file.read_text().splitlines() if line.strip()]
            names = [r["name"] for r in records]

            # Attention tensors must appear for both GatedDeltaNet and MHA layers
            attn_before = [n for n in names if "self_attn" in n and "[before]hidden_states" in n]
            attn_after  = [n for n in names if "self_attn" in n and "[after]outputs" in n]
            self.assertGreater(len(attn_before), 0,
                f"[compile] No self_attn before records. Got:\n" + "\n".join(names))
            self.assertGreater(len(attn_after),  0, "[compile] No self_attn after records.")

            # Decoder-layer before/after must appear
            layer_before = [n for n in names if re.search(r"layers\.\d+\]\[before\]hidden_states", n)]
            layer_after  = [n for n in names if re.search(r"layers\.\d+\]\[after\]hidden_states", n)]
            self.assertGreater(len(layer_before), 0, "[compile] No decoder-layer before records.")
            self.assertGreater(len(layer_after),  0, "[compile] No decoder-layer after records.")

            # MoE experts must appear
            experts_before = [n for n in names if "experts" in n and "[before]" in n]
            experts_after  = [n for n in names if "experts" in n and "[after]" in n]
            self.assertGreater(len(experts_before), 0, "[compile] No experts before records.")
            self.assertGreater(len(experts_after),  0, "[compile] No experts after records.")

        _reset_acc_prober()
        torch._dynamo.reset()


# ---------------------------------------------------------------------------
# MHA fullgraph=True + prober q/k norm capture test
# ---------------------------------------------------------------------------

class TestAccProberMHAFullgraph(DeterministicDDPTestCase):
    """Call MultiHeadAttention.forward directly (not the full model) and verify:

      1. MultiHeadAttention.forward is a compiled function (fullgraph=True).
      2. The prober's list.append inside record_tensor is NOT a graph break in
         this PyTorch version — Dynamo inlines it as a Python side effect within
         the fullgraph=True compilation. Consequently counters["graph_break"]
         stays at zero after the forward.
      3. Because list.append runs inline during compiled execution, q_norm /
         k_norm tensors ARE captured in the JSONL dump.

    Why no warm-up step: a prior non-profile forward would compile MHA.forward
    with _skip_flag=True (list.append dead-code-eliminated).  The subsequent
    guard failure on _skip_flag would re-use the old cached graph rather than
    recompiling, producing no records.
    """

    def test_mha_fullgraph_with_prober_dumps_qk_norm(self):
        import torch._dynamo.utils as _dynamo_utils
        from xtuner.v1.utils.compile import is_compiled_function

        self.create_pg("cuda")
        _reset_acc_prober()
        torch._dynamo.reset()

        # Build the tiny model only to (a) trigger class-level torch.compile on
        # MultiHeadAttention.forward and (b) borrow the rotary_emb module.
        config = _make_small_model_config()
        config.compile_cfg = True
        model = config.build().to(torch.bfloat16).cuda()

        # Layer 3 is the full_attention (DenseDecoderLayer) in the 4-layer config.
        mha = model.layers["3"].self_attn  # MultiHeadAttention with qk_norm=True

        # MHA.forward class attribute must be a compiled function (fullgraph=True).
        self.assertTrue(
            is_compiled_function(mha.forward),
            "MultiHeadAttention.forward should be a compiled function",
        )


        B, SEQ_LEN = 1, 16
        PROBE_STEP = 1

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Register prober ONLY on the MHA module (and its sub-modules q_norm, k_norm).
            ProberList.setup(Path(tmp_dir), [PROBE_STEP], ["AccProber"])
            register_prober_list(mha)

            ProberList.set_step(PROBE_STEP)
            ProberList.set_micro_batch_iter(0)

            # Build inputs for MHA.forward.
            hidden_states = torch.randn(
                B, SEQ_LEN, config.hidden_size, dtype=torch.bfloat16, device="cuda"
            )
            position_ids = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0)
            with torch.no_grad():
                # Use the model's shared rotary_emb for correctly-shaped cos/sin.
                cos, sin = model.rotary_emb(hidden_states, position_ids)

            input_ids = torch.randint(0, config.vocab_size, (B, SEQ_LEN), device="cuda")
            seq_ctx = SequenceContext.from_input_ids(input_ids=(input_ids,))

            _dynamo_utils.counters.clear()

            with torch.no_grad():
                mha(hidden_states, position_embeddings=(cos, sin), seq_ctx=seq_ctx)

            # ---- 1. list.append is NOT a graph break in fullgraph=True ----
            # Dynamo inlines it as a Python side effect; no split points are created.
            graph_break_count = sum(_dynamo_utils.counters["graph_break"].values())
            self.assertEqual(
                graph_break_count, 0,
                f"Expected zero graph breaks: list.append is inlined by Dynamo. "
                f"Got: {dict(_dynamo_utils.counters['graph_break'])}",
            )

            AccProber.after_micro_iter_forward()

            dump_file = (
                Path(tmp_dir)
                / "acc_prober"
                / f"Step_{PROBE_STEP}_MicroIter_0_RANK_{dist.get_rank()}_forward_records.jsonl"
            )
            self.assertTrue(dump_file.exists(), f"Dump file not found: {dump_file}")

            records = [json.loads(line) for line in dump_file.read_text().splitlines() if line.strip()]
            names = [r["name"] for r in records]

            # ---- 2. q_norm / k_norm tensors must be captured ----
            q_norm_records = [n for n in names if "q_norm" in n]
            k_norm_records = [n for n in names if "k_norm" in n]

            self.assertGreater(
                len(q_norm_records), 0,
                "Expected q_norm records from inside MHA.forward. Got names:\n" + "\n".join(names),
            )
            self.assertGreater(
                len(k_norm_records), 0,
                "Expected k_norm records from inside MHA.forward. Got names:\n" + "\n".join(names),
            )

        _reset_acc_prober()
        torch._dynamo.reset()
