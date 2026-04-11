import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from xtuner._testing import DeterministicDDPTestCase, patch_hf_rms_norm
from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import MiMoDenseConfig, get_model_config_from_hf


MIMO_PATH = os.environ.get("MIMO_7B_RL_PATH", "/mnt/shared-storage-user/llmrazor-share/model/MiMo-7B-RL")


class TestMiMoDense(DeterministicDDPTestCase):
    def test_mimo_dense_run(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for MiMo dense accuracy test")
        device = "cuda"
        tol = 1e-2
        self.create_pg(device)

        hf_model = AutoModelForCausalLM.from_pretrained(
            MIMO_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda",
        )
        patch_hf_rms_norm(hf_model)
        tokenizer = AutoTokenizer.from_pretrained(MIMO_PATH, trust_remote_code=True)
        input_ids = tokenizer("请解释为什么天空是蓝色的。", return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            expected_loss = hf_model(input_ids=input_ids, labels=input_ids.clone()).loss

        del hf_model
        torch.cuda.empty_cache()

        with torch.device("meta"):
            cfg = MiMoDenseConfig.from_hf(MIMO_PATH)
            cfg.compile_cfg = False
            mimo_model = cfg.build().to(torch.bfloat16)

        shift_input_ids = input_ids[:, :-1]
        shifted_labels = input_ids[:, 1:]
        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids,))

        loss_cfg = CELossConfig()
        loss_ctx = loss_cfg.build(data={"shifted_labels": shifted_labels}, sp_mesh=None)
        assert loss_ctx is not None
        loss_ctx = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])[0]

        mimo_model.from_hf(MIMO_PATH)

        with torch.no_grad():
            output = mimo_model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx, "mtp": None})

        self.assertTrue(torch.allclose(output["loss"], expected_loss.to(output["loss"].dtype), atol=tol, rtol=tol))

    def test_mimo_mtp_detach_main_trunk(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for MiMo MTP test")
        device = "cuda"
        self.create_pg(device)

        tokenizer = AutoTokenizer.from_pretrained(MIMO_PATH, trust_remote_code=True)
        input_ids = tokenizer("用一句话说明什么是梯度下降。", return_tensors="pt").input_ids.to("cuda")
        shift_input_ids = input_ids[:, :-1]
        shifted_labels = input_ids[:, 1:]
        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids,))

        with torch.device("meta"):
            cfg = MiMoDenseConfig.from_hf(MIMO_PATH)
            cfg.compile_cfg = False
            mimo_model = cfg.build().to(torch.bfloat16)

        mimo_model.from_hf(MIMO_PATH)
        mimo_model.zero_grad(set_to_none=True)

        data_batch = [{"seq_ctx": seq_ctx, "shifted_labels": shifted_labels}]
        loss_ctx = mimo_model.build_loss_ctx_batch(data_batch, sp_mesh=None)[0]

        output = mimo_model(seq_ctx=seq_ctx, loss_ctx=loss_ctx)
        self.assertIsNotNone(output["mtp_loss"])
        assert output["mtp_loss"] is not None
        self.assertFalse(torch.isnan(output["mtp_loss"]))

        output["mtp_loss"].backward()

        mtp_grad = mimo_model.mtp_block.layers[0].input_proj.weight.grad
        self.assertIsNotNone(mtp_grad)
        assert mtp_grad is not None
        self.assertGreater(mtp_grad.abs().sum().item(), 0.0)

        last_dense_grad = mimo_model.layers[str(mimo_model.config.num_hidden_layers - 1)].self_attn.q_proj.weight.grad
        embed_grad = mimo_model.embed_tokens.weight.grad
        self.assertTrue(last_dense_grad is None or torch.allclose(last_dense_grad, torch.zeros_like(last_dense_grad)))
        self.assertTrue(embed_grad is None or torch.allclose(embed_grad, torch.zeros_like(embed_grad)))

    def test_save_hf_with_mtp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for MiMo save_hf test")
        device = "cuda"
        self.create_pg(device)

        with torch.device("meta"):
            cfg = get_model_config_from_hf(Path(MIMO_PATH))
            cfg.compile_cfg = False
            mimo_model = cfg.build().to(torch.bfloat16)

        fsdp_config = FSDPConfig(cpu_offload=False)
        mimo_model.fully_shard(fsdp_config=fsdp_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            syncdir = [tmpdir]
            dist.broadcast_object_list(syncdir, src=0)
            tmpdir = Path(syncdir[0])

            mimo_model.from_hf(MIMO_PATH)
            mimo_model.save_hf(tmpdir)

            if dist.get_rank() == 0:
                saved_index = json.loads((tmpdir / "model.safetensors.index.json").read_text())
                self.assertIn("model.mtp_layers.0.input_proj.weight", saved_index["weight_map"])
                self.assertIn("model.layers.0.self_attn.q_proj.bias", saved_index["weight_map"])

                origin_index = json.loads((Path(MIMO_PATH) / "model.safetensors.index.json").read_text())
                keys_to_check = [
                    "model.layers.0.self_attn.q_proj.bias",
                    "model.layers.0.self_attn.q_proj.weight",
                    "model.mtp_layers.0.input_proj.weight",
                    "model.mtp_layers.0.token_layernorm.weight",
                ]
                cache = {}
                for key in keys_to_check:
                    origin_name = origin_index["weight_map"][key]
                    saved_name = saved_index["weight_map"][key]

                    origin_path = Path(MIMO_PATH) / origin_name
                    saved_path = tmpdir / saved_name
                    if origin_path not in cache:
                        cache[origin_path] = safe_open(origin_path, framework="pt")
                    if saved_path not in cache:
                        cache[saved_path] = safe_open(saved_path, framework="pt")

                    self.assertTrue(
                        torch.equal(cache[origin_path].get_tensor(key), cache[saved_path].get_tensor(key))
                    )

                self.assertTrue((tmpdir / "configuration_mimo.py").exists())
                self.assertTrue((tmpdir / "modeling_mimo.py").exists())

        dist.barrier()

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "1"))
