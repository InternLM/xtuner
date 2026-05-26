import json
import os
import tempfile
import unittest
from pathlib import Path

import parametrize
import torch
import torch.distributed as dist
from packaging.version import Version
from safetensors import safe_open
from transformers import __version__ as transformers_version

from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model import Qwen3_5_VLDense4BConfig
from xtuner.v1.model.compose.qwen3_vl.modeling_vision import init_world_mesh
from xtuner.v1.model.dense.qwen3_5_text import Qwen3_5_VLTextDense4BConfig
from xtuner._testing import DeterministicDDPTestCase


# Qwen3.5-4B (dense, hybrid linear + full attention VLM)
QWEN3_5_DENSE_4B_PATH = os.environ["QWEN3_5_DENSE_4B_PATH"]


@unittest.skipIf(
    Version(transformers_version) < Version("5.9.0"),
    f"transformers >= 5.9.0 is required, but got {transformers_version}",
)
class TestQwen3_5_VLDense(DeterministicDDPTestCase):
    @parametrize.parametrize("device,layer_idx", [("cuda", 3), ("cuda", 0)])
    def test_decoder_layer_bitwise_parity(self, device, layer_idx):
        # One decoder layer (full=3 / linear=0) -> final norm -> lm_head -> CE loss -> backward.
        # Under XTUNER_HF_IMPL (eager) the layer output, the loss, and the input gradient dL/dx must all
        # match HF bitwise — i.e. forward AND backward parity at the layer level. Only this layer + norm
        # + lm_head are materialized/loaded on GPU (XTuner builds the rest on meta; HF builds a standalone
        # layer), so this runs at any model scale, including models too large to forward end-to-end.
        import torch.nn.functional as F
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

        from xtuner.v1.utils import HFCheckpointLoader

        self.create_pg(device)
        with self.hf_impl():
            loader = HFCheckpointLoader(QWEN3_5_DENSE_4B_PATH)

            # ---- XTuner: build the full tower on meta, materialize only layer/norm/lm_head ----
            with torch.device("meta"):
                cfg = Qwen3_5_VLTextDense4BConfig(compile_cfg=False)
                model = cfg.build()
            layer_type = cfg.layers_type[layer_idx]
            is_linear = layer_type == "linear_attention"
            xt_layer = model.layers[str(layer_idx)]
            self.materialize_submodule(model, xt_layer, loader)
            self.materialize_submodule(model, model.norm, loader)
            self.materialize_submodule(model, model.lm_head, loader)
            model.rotary_emb.to("cuda")  # built on CPU with real buffers even under meta; no checkpoint weights

            # ---- HF: meta-build the full compose, then materialize only the layer/norm/lm_head we
            # touch. Mirrors the XTuner side: `materialize_submodule` recovers each submodule's
            # checkpoint prefix from its `named_modules` path, and honors HF's
            # `_tied_weights_keys` so `lm_head.weight` is loaded from `embed_tokens.weight`.
            hf_cfg = Qwen3_5Config.from_pretrained(QWEN3_5_DENSE_4B_PATH)
            hf_cfg._attn_implementation = "eager"
            hf_cfg.text_config._attn_implementation = "eager"
            hf_cfg.vision_config._attn_implementation = "eager"
            with torch.device("meta"):
                hf_compose = Qwen3_5ForConditionalGeneration(hf_cfg).eval()
            hf_layer = hf_compose.model.language_model.layers[layer_idx]
            hf_norm = hf_compose.model.language_model.norm
            hf_lm_head = hf_compose.lm_head
            self.materialize_submodule(hf_compose, hf_layer, loader)
            self.materialize_submodule(hf_compose, hf_norm, loader)
            self.materialize_submodule(hf_compose, hf_lm_head, loader)

            seq = 16
            ids = torch.randint(0, 1000, (1, seq), device="cuda")
            seq_ctx = SequenceContext.from_input_ids(input_ids=(ids,))
            seq_ctx.to("cuda")
            cos, sin = model.rotary_emb(
                torch.empty(1, seq, cfg.hidden_size, device="cuda", dtype=torch.bfloat16), seq_ctx.position_ids
            )
            labels = torch.randint(0, cfg.vocab_size, (seq,), device="cuda")
            base = torch.randn(1, seq, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)

            # full attention needs a causal mask; the linear (GatedDeltaNet) layer ignores it.
            attn_mask = (
                None
                if is_linear
                else torch.triu(
                    torch.full((seq, seq), float("-inf"), device="cuda", dtype=torch.bfloat16), diagonal=1
                )[None, None]
            )

            x_hf = base.clone().requires_grad_(True)
            o_hf = hf_layer(x_hf, position_embeddings=(cos, sin), attention_mask=attn_mask)
            o_hf = o_hf[0] if isinstance(o_hf, tuple) else o_hf
            loss_hf = F.cross_entropy(hf_lm_head(hf_norm(o_hf)).reshape(-1, cfg.vocab_size), labels)
            loss_hf.backward()

            x_xt = base.clone().requires_grad_(True)
            o_xt = xt_layer(x_xt, (cos, sin), seq_ctx)
            loss_xt = F.cross_entropy(F.linear(model.norm(o_xt), model.lm_head.weight).reshape(-1, cfg.vocab_size), labels)
            loss_xt.backward()

        # forward (layer output), loss, and backward (dL/dx) must all be bitwise.
        out_diff = (o_hf.float() - o_xt.float().reshape(o_hf.shape)).abs().max().item()
        loss_diff = (loss_hf.float() - loss_xt.float()).abs().item()
        grad_diff = (x_hf.grad.float() - x_xt.grad.float()).abs().max().item()  # type: ignore[union-attr]
        self.assertEqual(out_diff, 0.0, f"layer {layer_idx} [{layer_type}] output not bitwise: max diff {out_diff}")
        self.assertEqual(loss_diff, 0.0, f"layer {layer_idx} [{layer_type}] loss not bitwise: {loss_diff}")
        self.assertEqual(grad_diff, 0.0, f"layer {layer_idx} [{layer_type}] dL/dx not bitwise: max diff {grad_diff}")
        dist.barrier()

    @parametrize.parametrize("device", [("cuda",)])
    def test_vision_tower_bitwise_parity(self, device):
        # Vision tower (patch_embed + blocks + merger) bitwise vs HF, eager on both sides. Loads ONLY the
        # vision tower on each side (standalone; weights are the checkpoint's `model.visual.*`), not the
        # full VLM, so it stays memory-light for large models. Vision attention is non-causal: under
        # XTUNER_HF_IMPL the vision tower routes to eager_attention with causal=False. Pos-embed patch
        # is required.
        self.create_pg(device)
        self._patch_fast_pos_embed_interpolate()
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

        from xtuner.v1.utils import HFCheckpointLoader

        raw_data = {
            "id": 3,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "tests/resource/mscoco_dog_000000319154.jpg", "image_wh": [375, 500]},
                        },
                        {"type": "text", "text": "<IMG_CONTEXT>\n描述图片"},
                    ],
                },
                {"role": "assistant", "content": "狗。"},
            ],
        }
        inputs = self._tokenize_qwen3vl(raw_data)
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        with self.hf_impl():
            loader = HFCheckpointLoader(QWEN3_5_DENSE_4B_PATH)

            # ---- HF: meta-build the full compose, then materialize only the vision tower. The
            # `Qwen3_5VisionRotaryEmbedding.inv_freq` buffer (persistent=False, init-computed) is
            # rebuilt inside `materialize_submodule` from its stored `dim`/`theta`. ----
            hf_cfg = Qwen3_5Config.from_pretrained(QWEN3_5_DENSE_4B_PATH)
            hf_cfg._attn_implementation = "eager"
            hf_cfg.text_config._attn_implementation = "eager"
            hf_cfg.vision_config._attn_implementation = "eager"
            with torch.device("meta"):
                hf_compose = Qwen3_5ForConditionalGeneration(hf_cfg).eval()
            hf_vision = hf_compose.model.visual
            self.materialize_submodule(hf_compose, hf_vision, loader)
            hf_pv = pixel_values.clone().requires_grad_(True)
            hf_merged = hf_vision(hf_pv, grid_thw=image_grid_thw).pooler_output
            hf_merged.sum().backward()

            # ---- XTuner: same flow as HF — meta-build the compose, then materialize only
            # vision_tower + multi_modal_projector. XTuner splits HF's `visual` into vision_tower
            # (patches -> merged hidden) + projector (merger MLP -> text dim); HF's pooler_output is
            # post-merger so both are needed. ----
            with torch.device("meta"):
                xt_compose = Qwen3_5_VLDense4BConfig(compile_cfg=False).build()
            xt_vision = xt_compose.vision_tower
            xt_projector = xt_compose.multi_modal_projector
            self.materialize_submodule(xt_compose, xt_vision, loader)
            self.materialize_submodule(xt_compose, xt_projector, loader)
            self.assertEqual(xt_vision.blocks[0].attn.attn_impl_func.__name__, "eager_attention")
            xt_pv = pixel_values.clone().requires_grad_(True)
            xt_merged, xt_deepstack = xt_vision(xt_pv, image_grid_thw)
            xt_merged, _ = xt_projector(xt_merged, xt_deepstack)
            xt_merged.sum().backward()

        out_diff = (hf_merged.float() - xt_merged.float().reshape(hf_merged.shape)).abs().max().item()
        grad_diff = (hf_pv.grad.float() - xt_pv.grad.float()).abs().max().item()  # type: ignore[union-attr]
        self.assertEqual(out_diff, 0.0, f"vision tower output not bitwise: max diff {out_diff}")
        self.assertEqual(grad_diff, 0.0, f"vision tower dL/d(pixel_values) not bitwise: max diff {grad_diff}")
        dist.barrier()

    @parametrize.parametrize("device", [("cuda",)])
    def test_vl_forward_parity(self, device):
        # Whole-model (compose VLM) forward + backward parity vs HF on an image prompt — the VLM is the
        # real model, so this is the end-to-end integration check across vision + projector + text.
        # Eager both sides (XTUNER_HF_IMPL + HF eager): forward logits + loss are bitwise; the backward
        # dL/d(pixel_values) — which backprops through the whole text tower + projector + vision — is
        # checked at a tolerance (bf16 grad accumulates over depth; the bitwise backward guarantee is
        # test_decoder_layer_bitwise_parity). Image only: the image case already runs the text tower, so
        # a text-only case would just re-check the text forward.
        import torch.nn.functional as F

        self.create_pg(device)
        self._patch_fast_pos_embed_interpolate()
        from transformers import Qwen3_5ForConditionalGeneration

        raw_data = {
            "id": 3,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "tests/resource/mscoco_dog_000000319154.jpg", "image_wh": [375, 500]},
                        },
                        {"type": "text", "text": "<IMG_CONTEXT>\n描述图片"},
                    ],
                },
                {"role": "assistant", "content": "狗是棕色的。"},
            ],
        }
        inputs = self._tokenize_qwen3vl(raw_data)
        input_ids, image_grid_thw, position_ids = inputs["input_ids"], inputs["image_grid_thw"], inputs["position_ids"]
        base_pixels = inputs["pixel_values"]

        with self.hf_impl():
            hf = Qwen3_5ForConditionalGeneration.from_pretrained(
                QWEN3_5_DENSE_4B_PATH, dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda"
            ).eval()
            pv_hf = base_pixels.clone().requires_grad_(True)
            hf_logits = hf(
                input_ids=input_ids,
                pixel_values=pv_hf,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                use_cache=False,
            ).logits
            labels = torch.randint(0, hf_logits.size(-1), (hf_logits.size(1),), device="cuda")
            loss_hf = F.cross_entropy(hf_logits.reshape(-1, hf_logits.size(-1)), labels)
            loss_hf.backward()
            del hf
            torch.cuda.empty_cache()

            with torch.device("meta"):
                model_cfg = Qwen3_5_VLDense4BConfig(compile_cfg=False)  # text + vision -> eager via env
                model = model_cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)
            model.from_hf(QWEN3_5_DENSE_4B_PATH)
            model.eval()

            pv_xt = base_pixels.clone().requires_grad_(True)
            seq_ctx = SequenceContext.from_input_ids(input_ids=(input_ids,))
            seq_ctx.to("cuda")
            seq_ctx.image_grid_thw = image_grid_thw
            seq_ctx.pixel_values = pv_xt
            if position_ids is not None:
                seq_ctx.position_ids = position_ids
            xt_logits = model(seq_ctx=seq_ctx, loss_ctx=None)["logits"]
            loss_xt = F.cross_entropy(xt_logits.reshape(-1, xt_logits.size(-1)), labels)
            loss_xt.backward()

        logit_diff = (hf_logits.float() - xt_logits.float().reshape(hf_logits.shape)).abs().max().item()
        loss_diff = (loss_hf.float() - loss_xt.float()).abs().item()
        self.assertEqual(logit_diff, 0.0, f"VL logits not bitwise-equal: max diff {logit_diff}")
        self.assertEqual(loss_diff, 0.0, f"VL loss not bitwise-equal: {loss_diff}")
        # Backward (dL/d(pixel_values)) backprops through CE -> lm_head -> 32-layer LM ->
        # masked_scatter -> projector -> 24-layer ViT. With `XTUNER_HF_IMPL` routing
        # `GatedDeltaNet` to fla's high-level `chunk_gated_delta_rule` + HF-style
        # `causal_conv1d_fn` (no XTuner custom_op wrap, no seq_idx-driven layout switch) and
        # the `XTUNER_DETERMINISTIC` Triton autotune pin (`tests/conftest.py`), every backward
        # op matches HF byte-for-byte — full e2e backward is bitwise.
        grad_diff = (pv_hf.grad.float() - pv_xt.grad.float()).abs().max().item()  # type: ignore[union-attr]
        self.assertEqual(grad_diff, 0.0, f"VL dL/d(pixel_values) not bitwise: max diff {grad_diff}")
        dist.barrier()

    @parametrize.parametrize("device", [("cuda",)])
    def test_model_forward_bitwise_reduced_layers(self, device):
        # Whole-model bitwise parity that runs the real `compose.forward` / `Dense.forward` orchestration
        # — embed_tokens, rotary_emb call site, the layer loop, `_prepare_llm_inputs` image-embed
        # injection, final norm + lm_head, return packing — none of which are exercised by
        # `test_decoder_layer_bitwise_parity` (that test inlines them by hand). To keep this runnable at
        # any model scale we truncate `text_config.num_hidden_layers` to N: per-layer numerical
        # correctness is already owned by the decoder-layer test, so this test does not need every
        # layer present — it only needs the forward orchestration to run end-to-end. N=4 covers both
        # `linear_attention` (idx 0-2) and `full_attention` (idx 3) under Qwen3.5's `(i+1)%4==0`
        # pattern. Vision tower / projector are not truncated. `XTUNER_HF_IMPL` + `XTUNER_DETERMINISTIC`
        # + HF eager → logits / loss / `dL/d(pixel_values)` all bitwise.
        import torch.nn.functional as F

        N_TEXT_LAYERS = 4

        self.create_pg(device)
        self._patch_fast_pos_embed_interpolate()
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

        raw_data = {
            "id": 3,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "tests/resource/mscoco_dog_000000319154.jpg", "image_wh": [375, 500]},
                        },
                        {"type": "text", "text": "<IMG_CONTEXT>\n描述图片"},
                    ],
                },
                {"role": "assistant", "content": "狗是棕色的。"},
            ],
        }
        inputs = self._tokenize_qwen3vl(raw_data)
        input_ids, image_grid_thw, position_ids = inputs["input_ids"], inputs["image_grid_thw"], inputs["position_ids"]
        base_pixels = inputs["pixel_values"]

        with self.hf_impl():
            # ---- HF: truncate `text_config.num_hidden_layers` then `from_pretrained(config=...)`. HF
            # silently ignores the unused layer 4..31 ckpt keys ("UNEXPECTED" warning) and skips the
            # meta+materialize song-and-dance — keeps the test's HF side close to how a user would
            # actually downscale a real model, and avoids the subtle lazy-init / tie_weights gotchas
            # that meta+materialize would otherwise need to special-case.
            hf_cfg = Qwen3_5Config.from_pretrained(QWEN3_5_DENSE_4B_PATH)
            hf_cfg._attn_implementation = "eager"
            hf_cfg.text_config._attn_implementation = "eager"
            hf_cfg.vision_config._attn_implementation = "eager"
            hf_cfg.text_config.num_hidden_layers = N_TEXT_LAYERS
            hf_cfg.text_config.layer_types = hf_cfg.text_config.layer_types[:N_TEXT_LAYERS]
            hf = Qwen3_5ForConditionalGeneration.from_pretrained(
                QWEN3_5_DENSE_4B_PATH, config=hf_cfg, dtype=torch.bfloat16, device_map="cuda"
            ).eval()

            pv_hf = base_pixels.clone().requires_grad_(True)
            hf_logits = hf(
                input_ids=input_ids,
                pixel_values=pv_hf,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                use_cache=False,
            ).logits
            labels = torch.randint(0, hf_logits.size(-1), (hf_logits.size(1),), device="cuda")
            loss_hf = F.cross_entropy(hf_logits.reshape(-1, hf_logits.size(-1)), labels)
            loss_hf.backward()
            del hf
            torch.cuda.empty_cache()

            # ---- XTuner: same truncation. `layers_type` is a computed_field over `num_hidden_layers`,
            # so updating one field is enough. compose.from_hf is strict=True by default; since the
            # truncated model only expects layer 0..N-1 keys (and those exist in the ckpt), the
            # missing-set is empty and load succeeds. The extra layer N..31 ckpt keys go unread.
            text_cfg = Qwen3_5_VLTextDense4BConfig().model_copy(update={"num_hidden_layers": N_TEXT_LAYERS})
            compose_cfg = Qwen3_5_VLDense4BConfig(text_config=text_cfg, compile_cfg=False)
            with torch.device("meta"):
                model = compose_cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)
            model.from_hf(QWEN3_5_DENSE_4B_PATH)
            model.eval()

            pv_xt = base_pixels.clone().requires_grad_(True)
            seq_ctx = SequenceContext.from_input_ids(input_ids=(input_ids,))
            seq_ctx.to("cuda")
            seq_ctx.image_grid_thw = image_grid_thw
            seq_ctx.pixel_values = pv_xt
            if position_ids is not None:
                seq_ctx.position_ids = position_ids
            xt_logits = model(seq_ctx=seq_ctx, loss_ctx=None)["logits"]
            loss_xt = F.cross_entropy(xt_logits.reshape(-1, xt_logits.size(-1)), labels)
            loss_xt.backward()

        logit_diff = (hf_logits.float() - xt_logits.float().reshape(hf_logits.shape)).abs().max().item()
        loss_diff = (loss_hf.float() - loss_xt.float()).abs().item()
        grad_diff = (pv_hf.grad.float() - pv_xt.grad.float()).abs().max().item()  # type: ignore[union-attr]
        self.assertEqual(logit_diff, 0.0, f"reduced-layer VL logits not bitwise: max diff {logit_diff}")
        self.assertEqual(loss_diff, 0.0, f"reduced-layer VL loss not bitwise: {loss_diff}")
        self.assertEqual(grad_diff, 0.0, f"reduced-layer VL dL/d(pixel_values) not bitwise: max diff {grad_diff}")
        dist.barrier()

    @parametrize.parametrize("device", [("cuda",)])
    def test_save_hf_round_trip(self, device):
        # MTP is deferred for the dense port, so the 15 ``mtp.*`` checkpoint keys are
        # neither loaded nor re-saved. The round-trip is therefore asserted over the
        # non-``mtp.*`` keys only, and the saved index must contain no ``mtp.*`` key.
        self.create_pg(device)

        with torch.device("meta"):
            model_cfg = Qwen3_5_VLDense4BConfig(compile_cfg=False)
            model = model_cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)

        fsdp_config = FSDPConfig(cpu_offload=False)
        fsdp_mesh = init_world_mesh()
        model.vision_tower.fsdp_mesh = fsdp_mesh
        model.vision_tower.fsdp_config = fsdp_config
        model.fully_shard(fsdp_config=fsdp_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            syncdir = [tmpdir]
            dist.broadcast_object_list(syncdir, src=0)
            tmpdir = Path(syncdir[0])
            model.from_hf(QWEN3_5_DENSE_4B_PATH)
            model.save_hf(tmpdir)

            origin_hf_path = Path(QWEN3_5_DENSE_4B_PATH)
            origin_index_path = origin_hf_path / "model.safetensors.index.json"
            saved_index_path = tmpdir / "model.safetensors.index.json"

            if dist.get_rank() == 0:
                with open(origin_index_path, "r") as f:
                    origin_index = json.load(f)
                with open(saved_index_path, "r") as f:
                    saved_index = json.load(f)

                cache_save_fh: dict = {}

                for key in origin_index["weight_map"].keys():
                    if key.startswith("mtp."):
                        self.assertNotIn(key, saved_index["weight_map"])
                        continue

                    origin_safetensor_name = origin_index["weight_map"][key]
                    saved_safetensor_name = saved_index["weight_map"][key]

                    origin_sf_fh_name = str(origin_hf_path / origin_safetensor_name)
                    saved_sf_fh_name = str(tmpdir / saved_safetensor_name)

                    if origin_sf_fh_name not in cache_save_fh:
                        cache_save_fh[origin_sf_fh_name] = safe_open(origin_sf_fh_name, framework="pt")
                    if saved_sf_fh_name not in cache_save_fh:
                        cache_save_fh[saved_sf_fh_name] = safe_open(saved_sf_fh_name, framework="pt")

                    origin_tensor = cache_save_fh[origin_sf_fh_name].get_tensor(key)
                    saved_tensor = cache_save_fh[saved_sf_fh_name].get_tensor(key)

                    self.assertTrue(torch.equal(origin_tensor, saved_tensor), f"Tensor mismatch for key: {key}")

                mtp_keys = [key for key in saved_index["weight_map"].keys() if key.startswith("mtp.")]
                self.assertListEqual(mtp_keys, [])

                safetensor_keys: list[str] = []
                for safetensor_path in tmpdir.glob("*.safetensors"):
                    fh = safe_open(str(safetensor_path), framework="pt")
                    safetensor_keys.extend(fh.keys())
                safetensor_keys.sort()
                model_index_keys = list(saved_index["weight_map"].keys())
                model_index_keys.sort()
                self.assertListEqual(safetensor_keys, model_index_keys)

        dist.barrier()

    def _patch_fast_pos_embed_interpolate(self) -> None:
        # HF's fast_pos_embed_interpolate returns fp32; the reused XTuner vision forward adds
        # pos_embeds without a cast, so cast the result back to the pos_embed dtype here to
        # avoid an fp32/bf16 LayerNorm mismatch.
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        from xtuner.v1.model.compose.qwen3_vl.modeling_vision import Qwen3VLVisionModel

        def _interp(self, grid_thw):
            return Qwen3_5VisionModel.fast_pos_embed_interpolate(self, grid_thw).to(self.pos_embed.weight.dtype)

        Qwen3VLVisionModel.fast_pos_embed_interpolate = _interp

    def _tokenize_qwen3vl(self, raw_data) -> dict:
        # Tokenize one Qwen3.5-VL image SFT sample and return device-resident model inputs.
        from transformers import AutoTokenizer

        from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig

        tokenizer = AutoTokenizer.from_pretrained(QWEN3_5_DENSE_4B_PATH)
        tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN3_5_DENSE_4B_PATH, add_vision_id=True).build(tokenizer)
        tokenized = tokenize_fn(raw_data)
        return {
            "input_ids": torch.tensor(tokenized["input_ids"])[None].cuda(),
            "labels": torch.tensor(tokenized["labels"])[None].cuda(),
            "pixel_values": tokenized["pixel_values"].cuda(),
            "image_grid_thw": tokenized["image_grid_thw"].cuda(),
            "position_ids": tokenized["position_ids"].cuda(),
        }

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "1"))
