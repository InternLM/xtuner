import os
import json

import parametrize
import torch
from torch.testing._internal.common_distributed import DistributedTestBase
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import tempfile
from pathlib import Path
from safetensors import safe_open

from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.dense.qwen3 import Qwen3_8BConfig
from xtuner.v1.config import FSDPConfig
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.loss import CELossContext

# Qwen3 8B
QWEN3_PATH = os.environ["QWEN3_PATH"]


class TestQwen3Dense(DistributedTestBase):
    @parametrize.parametrize(
        "device,tp_size,compile,tol,loss_class",
        [
            ("cuda", 1, False, 1e-2, "cross_entropy"),
            ("cuda", 1, False, 1e-2, "chunk_cross_entropy"),
        ],
    )
    def test_qwen3_dense_run(self, device, tp_size, compile, tol, loss_class):
        self.create_pg(device)
        if not compile:
            maybe_compile.clear_compile_targets()

        hf_model = AutoModelForCausalLM.from_pretrained(
            QWEN3_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_PATH)
        input_ids = tokenizer("吃葡萄不吐葡萄皮", return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            output = hf_model(
                input_ids=input_ids,
                labels=input_ids.clone(),
            )
        expected_loss = output.loss

        del hf_model
        torch.cuda.empty_cache()

        with torch.device("meta"):
            cfg = Qwen3_8BConfig()
            qwen_model = cfg.build().to(torch.bfloat16)

        shift_input_ids = input_ids[:, :-1]
        shift_labels = input_ids[:, 1:]
        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
        data_batch = [{'seq_ctx': seq_ctx, 'labels': shift_labels}]
        loss_ctx = CELossContext()
        data_batch = loss_ctx.build_list_ctx(data_batch, device='cuda')[0]
        qwen_model.from_hf(QWEN3_PATH)

        with torch.no_grad():
            output = qwen_model(
                seq_ctx=data_batch['seq_ctx'],
                loss_ctx=data_batch['loss_ctx'],
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=tol, rtol=tol))

    @parametrize.parametrize(
        "device,tp_size",
        [
            ("cuda", 1),
        ],
    )
    def test_fsdp_accuracy(self, device, tp_size):
        self.create_pg(device)
        maybe_compile.clear_compile_targets()
        hf_model = AutoModelForCausalLM.from_pretrained(
            QWEN3_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_PATH, trust_remote_code=True)
        input_ids = tokenizer("吃葡萄不吐葡萄皮", return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            output = hf_model(
                input_ids=input_ids,
                labels=input_ids.clone(),
            )
        expected_loss = output.loss

        del hf_model
        torch.cuda.empty_cache()

        with torch.device("meta"):
            cfg = Qwen3_8BConfig()
            qwen_model = cfg.build().to(torch.bfloat16)

        fsdp_config = FSDPConfig(
            tp_size=tp_size,
            cpu_offload=False,
        )

        shift_input_ids = input_ids[:, :-1]
        shift_labels = input_ids[:, 1:]
        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
        data_batch = [{'seq_ctx': seq_ctx, 'labels': shift_labels}]
        loss_ctx = CELossContext()
        data_batch = loss_ctx.build_list_ctx(data_batch, device='cuda')[0]
        qwen_model.fully_shard(fsdp_config=fsdp_config)
        qwen_model.from_hf(QWEN3_PATH)

        with torch.no_grad():
            output = qwen_model(
                seq_ctx=data_batch['seq_ctx'],
                loss_ctx=data_batch['loss_ctx'],
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=1e-2, rtol=1e-2))

    @parametrize.parametrize(
        "use_sliding_window, max_window_layers, sliding_window",
        [
            (False, 6, 1024),
            (True, 6, 1024),
            (True, 4, 2048),
        ],
    )
    def test_sliding_windows(self, use_sliding_window, max_window_layers, sliding_window):
        self.create_pg('cuda')
        # test param
        with torch.device("meta"):
            num_hidden_layers = 6
            attention = MHAConfig(num_attention_heads=32,
                                  num_key_value_heads=8,
                                  head_dim=128,
                                  qk_norm=True,
                                  sliding_window=sliding_window)
            cfg = Qwen3_8BConfig(num_hidden_layers=num_hidden_layers,
                                 use_sliding_window=use_sliding_window,
                                 max_window_layers=max_window_layers,
                                 attention=attention)
            qwen_model = cfg.build().to(torch.bfloat16)

        if use_sliding_window is False or max_window_layers >= num_hidden_layers:
            expected_sliding_window_size_list = [(-1, -1) for _ in range(num_hidden_layers)]
        else:
            expected_sliding_window_size_list = [(-1, -1) for _ in range(max_window_layers)]
            expected_sliding_window_size_list += [(sliding_window, sliding_window) for _ in range(num_hidden_layers - max_window_layers)]

        model_sliding_window_size_list = []
        for layer in qwen_model.layers.values():
            model_sliding_window_size_list.append(layer.self_attn.window_size)

        self.assertListEqual(model_sliding_window_size_list, expected_sliding_window_size_list)

        # test forward
        if use_sliding_window is True:
            with torch.device("meta"):
                num_hidden_layers = 6
                attention = MHAConfig(num_attention_heads=32,
                                      num_key_value_heads=8,
                                      head_dim=128,
                                      qk_norm=True,
                                      sliding_window=sliding_window)
                cfg = Qwen3_8BConfig(num_hidden_layers=num_hidden_layers,
                                     use_sliding_window=use_sliding_window,
                                     max_window_layers=max_window_layers,
                                     attention=attention)
                qwen_model = cfg.build().to(torch.bfloat16)

            fsdp_config = FSDPConfig()
            tokenizer = AutoTokenizer.from_pretrained(QWEN3_PATH, trust_remote_code=True)
            input_ids = tokenizer("吃葡萄不吐葡萄皮", return_tensors="pt").input_ids.to("cuda")
            shift_input_ids = input_ids[:, :-1]
            shift_labels = input_ids[:, 1:]
            seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
            data_batch = [{'seq_ctx': seq_ctx, 'labels': shift_labels}]
            loss_ctx = CELossContext()
            data_batch = loss_ctx.build_list_ctx(data_batch, device='cuda')[0]
            qwen_model.fully_shard(fsdp_config=fsdp_config)
            qwen_model.from_hf(QWEN3_PATH, strict=False)

            with torch.no_grad():
                output = qwen_model(
                    seq_ctx=data_batch['seq_ctx'],
                    loss_ctx=data_batch['loss_ctx'],
                )
            assert "loss" in output

    @parametrize.parametrize(
        "device,tp_size",
        [
            ("cuda", 1),
        ],
    )
    def test_save_hf(self, device, tp_size):
        self.create_pg(device)
        with torch.device("meta"):
            cfg = Qwen3_8BConfig()
            qwen_model = cfg.build().to(torch.bfloat16)

        fsdp_config = FSDPConfig(
            tp_size=tp_size,
            cpu_offload=False,
        )

        cache_save_fh = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            syncdir = [tmpdir]
            dist.broadcast_object_list(syncdir, src=0)
            tmpdir = Path(syncdir[0])
            qwen_model.fully_shard(fsdp_config=fsdp_config)
            qwen_model.from_hf(QWEN3_PATH)
            qwen_model.save_hf(tmpdir)

            origin_hf_path = Path(QWEN3_PATH)
            origin_index_path = origin_hf_path / "model.safetensors.index.json"
            saved_index_path = tmpdir / "model.safetensors.index.json"

            # Test saved hf tensor value match the origin hf tensor value
            if dist.get_rank() == 0:
                with open(origin_index_path, "r") as f:
                    origin_index = json.load(f)
                with open(saved_index_path, "r") as f:
                    saved_index = json.load(f)

                for key in origin_index["weight_map"].keys():
                    origin_safetensor_name = origin_index["weight_map"][key]
                    saved_safetensor_name = saved_index["weight_map"][key]

                    origin_sf_fh_name = str(origin_hf_path / origin_safetensor_name)
                    expected_sf_fh_name = str(tmpdir / saved_safetensor_name)

                    if origin_safetensor_name not in cache_save_fh:
                        cache_save_fh[origin_safetensor_name] = safe_open(origin_sf_fh_name, framework="pt")
                    if saved_safetensor_name not in cache_save_fh:
                        cache_save_fh[saved_safetensor_name] = safe_open(expected_sf_fh_name, framework="pt")

                    origin_fh = cache_save_fh[origin_safetensor_name]
                    saved_fh = cache_save_fh[saved_safetensor_name]

                    origin_tensor = origin_fh.get_tensor(key)
                    saved_tensor = saved_fh.get_tensor(key)
                    self.assertTrue(torch.equal(origin_tensor, saved_tensor))

                # Test the tensor number in safetensors match the tensor number in model index
                safetensor_keys = []
                for safetensor_path in tmpdir.glob("*.safetensors"):
                    fh = cache_save_fh[safetensor_path.name]
                    safetensor_keys.extend(fh.keys())
                    safetensor_keys.sort()
                model_index_keys = list(saved_index["weight_map"].keys())
                model_index_keys.sort()

                self.assertListEqual(safetensor_keys, model_index_keys)
        dist.barrier()

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
