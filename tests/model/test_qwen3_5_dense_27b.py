import os
import unittest

import parametrize
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version
from transformers import __version__ as transformers_version

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import Qwen3_5_VLDense27BConfig
from xtuner.v1.model.dense.qwen3_5_text import Qwen3_5_VLTextDense27BConfig, Qwen3_5_VLTextDenseConfig
from xtuner.v1.module.attention import GatedDeltaNetConfig, MHAConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.rope import RopeParametersConfig
from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.utils.test_utils import init_data_mesh


QWEN3_8_27B_PATH = os.environ.get("QWEN3_8_27B_PATH") or os.environ.get("QWEN3_5_DENSE_27B_PATH")


def _tiny_qwen3_8_text_config() -> Qwen3_5_VLTextDenseConfig:
    """Qwen3.8 hybrid layout (GDN + gated MHA + MTP) at a size that fits SP2."""
    return Qwen3_5_VLTextDenseConfig(
        vocab_size=32,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=1,
        num_hidden_layers=4,
        hidden_size=64,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
        compile_cfg=False,
        attention=MHAConfig(
            with_gate=True,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            qk_norm=True,
            rms_norm_eps=1e-6,
            rms_norm_type="zero_centered",
        ),
        linear_attention=GatedDeltaNetConfig(
            num_value_heads=4,
            num_key_heads=2,
            key_head_dim=16,
            value_head_dim=16,
            conv_kernel_dim=4,
            hidden_act="silu",
            rms_norm_eps=1e-6,
        ),
        rope_parameters_cfg=RopeParametersConfig(
            rope_theta=10000000.0,
            rope_type="qwen3_vl",
            mrope_section=[1, 0, 0],
            partial_rotary_factor=0.25,
        ),
        mtp_config=MTPConfig(num_layers=2, share_weights=True),
        lm_loss_cfg=CELossConfig(mode="eager"),
    )


class TestQwen3_5_VLDense27BConfig(unittest.TestCase):
    def test_published_dims(self):
        cfg = Qwen3_5_VLDense27BConfig()
        text = cfg.text_config
        self.assertEqual(text.num_hidden_layers, 64)
        self.assertEqual(text.hidden_size, 5120)
        self.assertEqual(text.intermediate_size, 17408)
        self.assertEqual(text.attention.num_attention_heads, 24)
        self.assertEqual(text.attention.num_key_value_heads, 4)
        self.assertTrue(text.attention.with_gate)
        self.assertEqual(text.linear_attention.num_value_heads, 48)
        self.assertEqual(text.linear_attention.num_key_heads, 16)
        self.assertFalse(text.tie_word_embeddings)
        self.assertEqual(cfg.vision_config.depth, 27)
        self.assertEqual(cfg.vision_config.hidden_size, 1152)
        self.assertEqual(cfg.projector_config.text_hidden_size, 5120)
        self.assertEqual(sum(t == "full_attention" for t in text.layers_type), 16)
        self.assertEqual(text.layers_type[3], "full_attention")
        self.assertEqual(text.layers_type[0], "linear_attention")

    def test_mtp_block_and_hf_key_mapping(self):
        from xtuner.v1.module.mtp import MTPConfig

        cfg = Qwen3_5_VLTextDense27BConfig(mtp_config=MTPConfig(num_layers=1))
        self.assertFalse(cfg.tie_word_embeddings)
        self.assertEqual(cfg.layers_type[-1], "full_attention")
        with torch.device("meta"):
            model = cfg.build()
        self.assertIsNotNone(model.mtp_block)
        self.assertEqual(len(model.mtp_block.layers), 1)
        mapping = {
            "mtp_block.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            "mtp_block.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            "mtp_block.layers.0.eh_proj.weight": "mtp.fc.weight",
            "mtp_block.layers.0.final_layernorm.weight": "mtp.norm.weight",
            "mtp_block.layers.0.decoder_layer.self_attn.q_proj.weight": "mtp.layers.0.self_attn.q_proj.weight",
            "lm_head.weight": "lm_head.weight",
        }
        for xtuner_key, hf_key in mapping.items():
            self.assertEqual(model.to_hf_key_list(xtuner_key), [hf_key], xtuner_key)

        with torch.device("meta"):
            model_off = Qwen3_5_VLTextDense27BConfig().build()
        self.assertIsNone(model_off.mtp_block)


@unittest.skipIf(
    Version(transformers_version) < Version("5.8.0"),
    f"transformers >= 5.8.0 is required, but got {transformers_version}",
)
@unittest.skipIf(not QWEN3_8_27B_PATH, "QWEN3_8_27B_PATH or QWEN3_5_DENSE_27B_PATH is not set")
class TestQwen3_8_VLDense27B(DeterministicDDPTestCase):
    def _load_standalone_hf(self, module: nn.Module, loader, prefix: str) -> None:
        """Load an HF module from explicit checkpoint keys. 27B is untied, so lm_head
        must come from ``lm_head.weight``, not ``embed_tokens``."""
        module.to_empty(device="cuda")
        module.to(torch.bfloat16)
        self.load_params_from_hf(module, loader, key_for=prefix)

    @parametrize.parametrize("device,layer_idx", [("cuda", 3), ("cuda", 0)])
    def test_decoder_layer_bitwise_parity(self, device, layer_idx):
        from transformers import Qwen3_5Config
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm

        from xtuner.v1.utils import HFCheckpointLoader

        self.create_pg(device)
        with self.hf_impl():
            loader = HFCheckpointLoader(QWEN3_8_27B_PATH)

            with torch.device("meta"):
                cfg = Qwen3_5_VLTextDense27BConfig(compile_cfg=False)
                model = cfg.build()
            self.assertFalse(cfg.tie_word_embeddings)
            layer_type = cfg.layers_type[layer_idx]
            is_linear = layer_type == "linear_attention"
            xt_layer = model.layers[str(layer_idx)]
            self.materialize_submodule(model, xt_layer, loader)
            self.materialize_submodule(model, model.norm, loader)
            self.materialize_submodule(model, model.lm_head, loader)
            model.rotary_emb.to("cuda")

            hf_cfg = Qwen3_5Config.from_pretrained(QWEN3_8_27B_PATH)
            text_cfg = hf_cfg.text_config
            text_cfg._attn_implementation = "eager"
            self.assertFalse(text_cfg.tie_word_embeddings)
            with torch.device("meta"):
                hf_layer = Qwen3_5DecoderLayer(text_cfg, layer_idx).eval()
                hf_norm = Qwen3_5RMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps)
                hf_lm_head = nn.Linear(text_cfg.hidden_size, text_cfg.vocab_size, bias=False)
            self._load_standalone_hf(hf_layer, loader, f"model.language_model.layers.{layer_idx}.")
            self._load_standalone_hf(hf_norm, loader, "model.language_model.norm.")
            self._load_standalone_hf(hf_lm_head, loader, "lm_head.")

            self.assertEqual(
                (model.lm_head.weight.float() - hf_lm_head.weight.float()).abs().max().item(),
                0.0,
                "27B lm_head is untied; both sides must load lm_head.weight",
            )

            seq = 16
            ids = torch.randint(0, 1000, (1, seq), device="cuda")
            seq_ctx = SequenceContext.from_input_ids(input_ids=(ids,))
            seq_ctx.to("cuda")
            cos, sin = model.rotary_emb(
                torch.empty(1, seq, cfg.hidden_size, device="cuda", dtype=torch.bfloat16), seq_ctx.position_ids
            )
            labels = torch.randint(0, cfg.vocab_size, (seq,), device="cuda")
            base = torch.randn(1, seq, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)

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
            o_xt = xt_layer(x_xt, position_embeddings=(cos, sin), seq_ctx=seq_ctx)["hidden_states"]
            loss_xt = F.cross_entropy(
                F.linear(model.norm(o_xt), model.lm_head.weight).reshape(-1, cfg.vocab_size), labels
            )
            loss_xt.backward()

        out_diff = (o_hf.float() - o_xt.float().reshape(o_hf.shape)).abs().max().item()
        loss_diff = (loss_hf.float() - loss_xt.float()).abs().item()
        grad_diff = (x_hf.grad.float() - x_xt.grad.float()).abs().max().item()  # type: ignore[union-attr]
        self.assertEqual(out_diff, 0.0, f"layer {layer_idx} [{layer_type}] output not bitwise: max diff {out_diff}")
        self.assertEqual(loss_diff, 0.0, f"layer {layer_idx} [{layer_type}] loss not bitwise: {loss_diff}")
        self.assertEqual(grad_diff, 0.0, f"layer {layer_idx} [{layer_type}] dL/dx not bitwise: max diff {grad_diff}")
        dist.barrier()

    def test_mtp_block_bitwise_parity(self):
        """Isolated MTP vs HF DecoderLayer/RMSNorm/Linear from ``mtp.*`` keys.

        transformers Qwen3.5 ignores ``mtp.*`` and has no MTP training forward, so
        this does not run a 64-layer 27B e2e. Only embed, rotary, and ``mtp_block``
        are materialized; the HF side is the published MTP graph (norm-embed,
        norm-hidden, concat, fc, full-attention decoder, final norm).
        """
        from transformers import Qwen3_5Config
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm

        from xtuner.v1.module.mtp import MTPConfig
        from xtuner.v1.module.mtp.utils import roll_sequence_context
        from xtuner.v1.utils import HFCheckpointLoader

        self.create_pg("cuda")
        with self.hf_impl():
            loader = HFCheckpointLoader(QWEN3_8_27B_PATH)

            with torch.device("meta"):
                cfg = Qwen3_5_VLTextDense27BConfig(mtp_config=MTPConfig(num_layers=1), compile_cfg=False)
                model = cfg.build()
            self.assertIsNotNone(model.mtp_block)
            self.materialize_submodule(model, model.embed_tokens, loader)
            self.materialize_submodule(model, model.mtp_block, loader)
            self.materialize_submodule(model, model.lm_head, loader)
            model.rotary_emb.to("cuda")

            hf_cfg = Qwen3_5Config.from_pretrained(QWEN3_8_27B_PATH)
            text_cfg = hf_cfg.text_config
            text_cfg._attn_implementation = "eager"
            full_idx = next(i for i, t in enumerate(text_cfg.layer_types) if t == "full_attention")
            with torch.device("meta"):
                hf_mtp_layer = Qwen3_5DecoderLayer(text_cfg, full_idx).eval()
                hf_enorm = Qwen3_5RMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps)
                hf_hnorm = Qwen3_5RMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps)
                hf_mtp_norm = Qwen3_5RMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps)
                hf_fc = nn.Linear(text_cfg.hidden_size * 2, text_cfg.hidden_size, bias=False)
                hf_embed = nn.Embedding(text_cfg.vocab_size, text_cfg.hidden_size, text_cfg.pad_token_id)
                hf_lm_head = nn.Linear(text_cfg.hidden_size, text_cfg.vocab_size, bias=False)
            self._load_standalone_hf(hf_mtp_layer, loader, "mtp.layers.0.")
            self._load_standalone_hf(hf_enorm, loader, "mtp.pre_fc_norm_embedding.")
            self._load_standalone_hf(hf_hnorm, loader, "mtp.pre_fc_norm_hidden.")
            self._load_standalone_hf(hf_mtp_norm, loader, "mtp.norm.")
            self._load_standalone_hf(hf_fc, loader, "mtp.fc.")
            self._load_standalone_hf(hf_embed, loader, "model.language_model.embed_tokens.")
            self._load_standalone_hf(hf_lm_head, loader, "lm_head.")

            seq = 16
            ids = torch.randint(0, 1000, (1, seq), device="cuda")
            seq_ctx = SequenceContext.from_input_ids(input_ids=(ids,))
            seq_ctx.to("cuda")
            cos, sin = model.rotary_emb(
                torch.empty(1, seq, cfg.hidden_size, device="cuda", dtype=torch.bfloat16), seq_ctx.position_ids
            )
            labels = torch.randint(0, cfg.vocab_size, (seq,), device="cuda")
            base = torch.randn(1, seq, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
            attn_mask = torch.triu(
                torch.full((seq, seq), float("-inf"), device="cuda", dtype=torch.bfloat16), diagonal=1
            )[None, None]

            x_hf = base.clone().requires_grad_(True)
            rolled = roll_sequence_context(seq_ctx, shifts=-1)
            fut = hf_embed(rolled.input_ids)
            projected = hf_fc(torch.cat([hf_enorm(fut), hf_hnorm(x_hf)], dim=-1))
            o_hf = hf_mtp_layer(projected, position_embeddings=(cos, sin), attention_mask=attn_mask)
            o_hf = o_hf[0] if isinstance(o_hf, tuple) else o_hf
            o_hf = hf_mtp_norm(o_hf)
            loss_hf = F.cross_entropy(hf_lm_head(o_hf).reshape(-1, cfg.vocab_size), labels)
            loss_hf.backward()

            x_xt = base.clone().requires_grad_(True)
            o_xt = model.mtp_block(
                x_xt,
                embed_tokens_fn=model.embed_tokens,
                position_embeddings=(cos, sin),
                seq_ctx=seq_ctx,
            )[0]["hidden_states"]
            loss_xt = F.cross_entropy(
                F.linear(o_xt, model.lm_head.weight).reshape(-1, cfg.vocab_size), labels
            )
            loss_xt.backward()

        out_diff = (o_hf.float() - o_xt.float().reshape(o_hf.shape)).abs().max().item()
        loss_diff = (loss_hf.float() - loss_xt.float()).abs().item()
        grad_diff = (x_hf.grad.float() - x_xt.grad.float()).abs().max().item()  # type: ignore[union-attr]
        self.assertEqual(out_diff, 0.0, f"mtp output not bitwise: max diff {out_diff}")
        self.assertEqual(loss_diff, 0.0, f"mtp loss not bitwise: {loss_diff}")
        self.assertEqual(grad_diff, 0.0, f"mtp dL/dx not bitwise: max diff {grad_diff}")
        dist.barrier()

    def _patch_fast_pos_embed_interpolate(self) -> None:
        from transformers.vision_utils import get_vision_bilinear_indices_and_weights

        from xtuner.v1.model.compose.qwen3_vl.modeling_vision import Qwen3VLVisionModel

        def _interp(self, grid_thw):
            indices, weights = get_vision_bilinear_indices_and_weights(
                grid_thw,
                num_grid_per_side=self.num_grid_per_side,
                spatial_merge_size=self.config.spatial_merge_size,
            )
            return (self.pos_embed(indices) * weights[:, :, None]).sum(0).to(self.pos_embed.weight.dtype)

        Qwen3VLVisionModel.fast_pos_embed_interpolate = _interp

    def _tokenize_qwen3vl(self, raw_data: dict) -> dict:
        from transformers import AutoTokenizer

        from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig

        tokenizer = AutoTokenizer.from_pretrained(QWEN3_8_27B_PATH)
        tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN3_8_27B_PATH, add_vision_id=True).build(tokenizer)
        tokenized = tokenize_fn(raw_data)
        return {
            "input_ids": torch.tensor(tokenized["input_ids"])[None].cuda(),
            "labels": torch.tensor(tokenized["labels"])[None].cuda(),
            "pixel_values": tokenized.get("pixel_values"),
            "image_grid_thw": tokenized.get("image_grid_thw"),
            "position_ids": tokenized.get("position_ids"),
        }

    def _full_model_loss(self, model, sample_type: str, device: str, sp_size: int) -> torch.Tensor:
        from transformers import Qwen3_5ForConditionalGeneration

        from xtuner.v1.data_proto.utils import pad_to_multiple_of
        from xtuner.v1.utils.test_utils import init_data_mesh

        if sample_type == "image":
            raw_data = {
                "id": 3,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "tests/resource/mscoco_dog_000000319154.jpg",
                                    "image_wh": [375, 500],
                                },
                            },
                            {"type": "text", "text": "<IMG_CONTEXT>\n描述图片"},
                        ],
                    },
                    {"role": "assistant", "content": "狗是棕色的。"},
                ],
            }
        else:
            raw_data = {
                "id": 3,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Translate this into chinese: Where my eyes gaze, only memories remain.",
                            }
                        ],
                    },
                    {"role": "assistant", "content": "目之所及，唯余旧忆。"},
                ],
            }

        inputs = self._tokenize_qwen3vl(raw_data)
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        if sample_type == "image":
            pixel_values = inputs["pixel_values"].cuda()
            image_grid_thw = inputs["image_grid_thw"].cuda()
            position_ids = inputs["position_ids"].cuda()
        else:
            pixel_values = None
            image_grid_thw = None
            position_ids = None

        if isinstance(model, Qwen3_5ForConditionalGeneration):
            with torch.no_grad():
                output = model(
                    input_ids=input_ids,
                    labels=labels,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    position_ids=position_ids,
                    use_cache=False,
                )
            dist.all_reduce(output.loss.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
            return output.loss

        shift_input_ids = pad_to_multiple_of(input_ids[:, :-1], padding_value=0, multiple_of=sp_size)
        shifted_labels = pad_to_multiple_of(labels[:, 1:], padding_value=-100, multiple_of=sp_size)
        if position_ids is not None:
            position_ids = position_ids[..., :-1]

        sp_mesh = None
        if sp_size > 1:
            data_mesh = init_data_mesh(device, sp_size=sp_size)
            sp_mesh = data_mesh["sp"]

        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to("cuda"),))
        seq_ctx.image_grid_thw = image_grid_thw
        seq_ctx.pixel_values = pixel_values
        if position_ids is not None:
            seq_ctx.position_ids = position_ids
        seq_ctx.to("cuda")
        if sp_size > 1:
            seq_ctx = seq_ctx.split(sp_mesh)

        data_batch = [{"seq_ctx": seq_ctx, "shifted_labels": shifted_labels}]
        loss_ctx = model.build_loss_ctx_batch(data_batch, sp_mesh=sp_mesh)[0]
        with torch.no_grad():
            output = model(seq_ctx=seq_ctx, loss_ctx=loss_ctx)
        return output["loss"]

    @parametrize.parametrize("device,sp_size,tol", [("cuda", 1, 2e-2)])
    def test_vl_full_model_loss_parity(self, device, sp_size, tol):
        """64-layer 27B VLM CE loss vs HuggingFace (text + image).

        transformers has no MTP training graph, so this checks the published
        backbone only. Flash attention on both sides, same 2e-2 band as
        Qwen3.5 35B-A3B. 27B dense already fills a 140GB GPU after the HF then
        XTuner loads, so this does not re-wrap with FSDP.
        """
        from transformers import Qwen3_5ForConditionalGeneration

        self.create_pg(device)
        self._patch_fast_pos_embed_interpolate()

        hf_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            QWEN3_8_27B_PATH,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        ).eval()
        dist.barrier()
        loss_hf_text = self._full_model_loss(hf_model, "text", device, sp_size)
        loss_hf_image = self._full_model_loss(hf_model, "image", device, sp_size)
        del hf_model
        torch.cuda.empty_cache()

        with torch.device("meta"):
            model_cfg = Qwen3_5_VLDense27BConfig(compile_cfg=False)
            model = model_cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)
        model.from_hf(QWEN3_8_27B_PATH)
        model.eval()

        loss_xt_text = self._full_model_loss(model, "text", device, sp_size)
        loss_xt_image = self._full_model_loss(model, "image", device, sp_size)
        self.assertTrue(
            torch.allclose(loss_xt_text, loss_hf_text.to(loss_xt_text.dtype), atol=tol, rtol=tol),
            f"Text loss mismatch: XTuner={loss_xt_text.item()}, HF={loss_hf_text.item()}",
        )
        self.assertTrue(
            torch.allclose(loss_xt_image, loss_hf_image.to(loss_xt_image.dtype), atol=tol, rtol=tol),
            f"Image loss mismatch: XTuner={loss_xt_image.item()}, HF={loss_hf_image.item()}",
        )
        dist.barrier()

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "1"))


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires 2 CUDA devices")
class TestQwen3_8MTPSequenceParallel(DeterministicDDPTestCase):
    def test_mtp_loss_and_gradients_match_full_sequence(self):
        """Packed LM/MTP loss and parameter grads under SP2 match the unsplit baseline."""
        self.create_pg("cuda")
        config = _tiny_qwen3_8_text_config()
        self.assertEqual(config.layers_type[-1], "full_attention")

        torch.manual_seed(17)
        baseline_model = config.build().to(device="cuda", dtype=torch.bfloat16)
        baseline_model.init_weights()
        sp_model = config.build().to(device="cuda", dtype=torch.bfloat16)
        sp_model.load_state_dict(baseline_model.state_dict())
        baseline_model.train()
        sp_model.train()

        sequence_0 = torch.tensor([[2, 3, 4, 5, 6, 7]], device="cuda")
        sequence_1 = torch.tensor([[8, 9, 10, 11]], device="cuda")
        packed_inputs = (sequence_0[:, :-1], sequence_1[:, :-1])
        shifted_labels = torch.cat((sequence_0[:, 1:], sequence_1[:, 1:]), dim=1)

        baseline_seq_ctx = SequenceContext.from_input_ids(packed_inputs, device="cuda")
        baseline_data = {"seq_ctx": baseline_seq_ctx, "shifted_labels": shifted_labels}
        baseline_loss_ctx = baseline_model.build_loss_ctx_batch([baseline_data], sp_mesh=None)[0]
        baseline_output = baseline_model(seq_ctx=baseline_seq_ctx, loss_ctx=baseline_loss_ctx)
        self.assertIsNotNone(baseline_output["mtp_loss"])
        (baseline_output["loss"] + baseline_output["mtp_loss"]).backward()

        baseline_gradients = {}
        for name, parameter in baseline_model.named_parameters():
            if parameter.grad is not None:
                gradient = parameter.grad.detach().float().clone()
                dist.all_reduce(gradient)
                baseline_gradients[name] = gradient / dist.get_world_size()

        sp_mesh = init_data_mesh("cuda", sp_size=2)["sp"]
        full_sp_seq_ctx = SequenceContext.from_input_ids(packed_inputs, device="cuda")
        sp_data = {"seq_ctx": full_sp_seq_ctx, "shifted_labels": shifted_labels}
        sp_loss_ctx = sp_model.build_loss_ctx_batch([sp_data], sp_mesh=sp_mesh)[0]
        sp_seq_ctx = full_sp_seq_ctx.split(sp_mesh)
        sp_output = sp_model(seq_ctx=sp_seq_ctx, loss_ctx=sp_loss_ctx)
        self.assertIsNotNone(sp_output["mtp_loss"])
        (sp_output["loss"] + sp_output["mtp_loss"]).backward()

        sp_gradients = {}
        for name, parameter in sp_model.named_parameters():
            if parameter.grad is not None:
                gradient = parameter.grad.detach().float().clone()
                dist.all_reduce(gradient)
                sp_gradients[name] = gradient / dist.get_world_size()

        torch.testing.assert_close(sp_output["loss"], baseline_output["loss"])
        torch.testing.assert_close(sp_output["mtp_loss"], baseline_output["mtp_loss"])
        self.assertEqual(sp_gradients.keys(), baseline_gradients.keys())
        max_relative_error = max(
            float((sp_gradients[name] - expected).norm() / expected.norm().clamp_min(1e-12))
            for name, expected in baseline_gradients.items()
        )
        min_cosine_similarity = min(
            float(
                F.cosine_similarity(
                    sp_gradients[name].flatten(),
                    expected.flatten(),
                    dim=0,
                )
            )
            for name, expected in baseline_gradients.items()
        )
        self.assertLess(max_relative_error, 2e-2)
        self.assertGreater(min_cosine_similarity, 0.9999)
        dist.barrier()

    @property
    def world_size(self) -> int:
        return 2
