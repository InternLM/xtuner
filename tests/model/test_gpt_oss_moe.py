import os
from functools import wraps
import torch.distributed as dist
from safetensors import safe_open
import json

import parametrize
import torch
from xtuner._testing import DeterministicDDPTestCase, patch_hf_rms_norm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import tempfile
from pathlib import Path
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.model.moe.gpt_oss import GptOss21BA3P6Config
from xtuner.v1.config import FSDPConfig
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContextInputItem

GPT_OSS_MINI_PATH = os.environ["GPT_OSS_MINI_PATH"]


def prepare(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        self.temp_dir = tempfile.TemporaryDirectory()
        ret = fn(self, *args, **kwargs)
        self.temp_dir.cleanup()
        return ret

    return wrapper


class TestGptOss(DeterministicDDPTestCase):
    @parametrize.parametrize(
        "device,dispatcher,ep_size,compile,tol,loss_class",
        [
            ("cuda", "all2all", 8, False, 1e-2, "cross_entropy"),
            ("cuda", None, 1, False, 1e-2, "cross_entropy"),
            # ("cuda", None, 1, False, 1e-2, "chunk_cross_entropy"),
        ],
    )
    @prepare
    def test_gpt_oss_run(self, device, dispatcher, ep_size, compile, tol, loss_class):
        os.environ["TRITON_CACHE_DIR"] = str(Path(self.temp_dir.name) / "triton_cache")
        self.create_pg(device)

        hf_config = AutoConfig.from_pretrained(GPT_OSS_MINI_PATH)
        hf_config.rope_scaling = None

        hf_model = AutoModelForCausalLM.from_pretrained(
            GPT_OSS_MINI_PATH,
            dtype=torch.bfloat16,
            config=hf_config,
            device_map="cuda"
        )
        hf_model.train()
        patch_hf_rms_norm((hf_model))
        tokenizer = AutoTokenizer.from_pretrained(GPT_OSS_MINI_PATH)
        input_ids = tokenizer("吃葡萄不吐葡萄皮", return_tensors="pt").input_ids.to("cuda")
        # assert input_ids.size(1) > 128
        with torch.no_grad():
            output = hf_model(
                input_ids=input_ids,
                labels=input_ids.clone()
            )
        expected_loss = output.loss

        del hf_model
        torch.cuda.empty_cache()

        with torch.device("meta"):
            cfg = GptOss21BA3P6Config(compile_cfg=False)
            cfg.dispatcher = dispatcher
            cfg.ep_size = ep_size
            gpt_oss_model = cfg.build().to(torch.bfloat16)

        shift_input_ids = input_ids[:, :-1]
        shifted_labels = input_ids[:, 1:]
        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
        loss_cfg = CELossConfig()
        seq_ctx_list = [seq_ctx]
        loss_ctx_input_list: list[CELossContextInputItem] = [CELossContextInputItem(shifted_labels=shifted_labels)]
        LossContext = loss_cfg.loss_ctx_cls
        batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
            loss_ctx_input_list, 
            loss_cfg,
        )
        loss_kwargs = batches_loss_kwargs[0]
        loss_ctx = LossContext(loss_cfg, loss_kwargs)
        seq_ctx = seq_ctx_list[0]
        gpt_oss_model.from_hf(GPT_OSS_MINI_PATH)
        with torch.no_grad():
            output = gpt_oss_model(
                seq_ctx=seq_ctx,
                loss_ctx=loss_ctx,
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=tol, rtol=tol))

    @parametrize.parametrize(
        "device,dispatcher,ep_size",
        [
            ("cuda", "all2all", 4),
            ("cuda", None, 1),
        ],
    )
    def test_fsdp_accuracy(self, device, dispatcher, ep_size):
        self.create_pg(device)

        hf_config = AutoConfig.from_pretrained(GPT_OSS_MINI_PATH)
        hf_config.rope_scaling = None
        hf_model = AutoModelForCausalLM.from_pretrained(
            GPT_OSS_MINI_PATH,
            dtype=torch.bfloat16,
            config=hf_config,
            device_map="cuda"
        )
        patch_hf_rms_norm((hf_model))
        hf_model.train()
        tokenizer = AutoTokenizer.from_pretrained(GPT_OSS_MINI_PATH)
        input_ids = tokenizer("吃葡萄不吐葡萄皮", return_tensors="pt").input_ids.to("cuda")
        # assert input_ids.size(1) > 128
        with torch.no_grad():
            output = hf_model(
                input_ids=input_ids,
                labels=input_ids.clone(),
            )
        expected_loss = output.loss

        del hf_model
        torch.cuda.empty_cache()

        with torch.device("meta"):
            cfg = GptOss21BA3P6Config(compile_cfg=False)
            cfg.ep_size = ep_size
            cfg.dispatcher = dispatcher
            gpt_oss_model = cfg.build().to(torch.bfloat16)

        fsdp_config = FSDPConfig(
            ep_size=ep_size,
            cpu_offload=False,
        )

        shift_input_ids = input_ids[:, :-1]
        shifted_labels = input_ids[:, 1:]
        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
        loss_cfg = CELossConfig()
        seq_ctx_list = [seq_ctx]
        loss_ctx_input_list: list[CELossContextInputItem] = [CELossContextInputItem(shifted_labels=shifted_labels)]
        LossContext = loss_cfg.loss_ctx_cls
        batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
            loss_ctx_input_list, 
            loss_cfg,
        )
        loss_kwargs = batches_loss_kwargs[0]
        loss_ctx = LossContext(loss_cfg, loss_kwargs)
        seq_ctx = seq_ctx_list[0]
        gpt_oss_model.fully_shard(fsdp_config=fsdp_config)
        gpt_oss_model.from_hf(GPT_OSS_MINI_PATH)

        with torch.no_grad():
            output = gpt_oss_model(
                seq_ctx=seq_ctx,
                loss_ctx=loss_ctx,
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=1e-2, rtol=1e-2))

    @parametrize.parametrize(
        "device,dispatcher,ep_size",
        [
            ("cuda", None, 1),
            ("cuda", "all2all", 4),
        ],
    )
    def test_save_hf(self, device, dispatcher, ep_size):
        self.create_pg(device)
        with torch.device("meta"):
            cfg = GptOss21BA3P6Config()
            cfg.dispatcher = dispatcher
            cfg.ep_size = ep_size
            gpt_oss_model = cfg.build().to(torch.bfloat16)

        fsdp_config = FSDPConfig(
            ep_size=ep_size,
            cpu_offload=False,
        )

        cache_save_fh = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            syncdir = [tmpdir]
            dist.broadcast_object_list(syncdir, src=0)
            tmpdir = Path(syncdir[0])
            gpt_oss_model.fully_shard(fsdp_config=fsdp_config)
            gpt_oss_model.from_hf(GPT_OSS_MINI_PATH)
            gpt_oss_model.save_hf(tmpdir)

            origin_hf_path = Path(GPT_OSS_MINI_PATH)
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
