import os
import json

import parametrize
import torch
from torch.testing._internal.common_distributed import DistributedTestBase
import torch.distributed as dist
from transformers import  AutoTokenizer
import tempfile
from pathlib import Path
from safetensors import safe_open
from unittest import skipIf
import transformers
from packaging import version

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.dense.qwen3 import Qwen3Dense4BConfig
from xtuner.v1.model.compose.qwen3_vl import Qwen3VLDense4BConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.config import FSDPConfig, LRConfig, AdamWConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.engine.vision_compose_train_engine import VisionComposeTrainEngine
from torch.optim.lr_scheduler import LambdaLR
from xtuner.v1.utils import pad_to_max_length
from xtuner.v1.utils.device import get_device
from xtuner.v1.model.base import ModelItem

# Qwen3 4B
QWEN3_PATH = os.environ["QWEN3_4B_PATH"]
QWEN3_VL_DENSE_PATH = os.environ["QWEN3_VL_DENSE_PATH"]
DEVICE = get_device()


class TestQwen3Dense4B(DistributedTestBase):

    @parametrize.parametrize(
        "device,tp_size",
        [
            ("cuda", 1),
        ],
    )
    def test_tie_embedding(self, device, tp_size):
        pg = self.create_pg(device)
        dense_cfg = Qwen3Dense4BConfig()
        optim_cfg: AdamWConfig = AdamWConfig()
        lr_cfg: LRConfig = LRConfig(lr_min=1e-3)
        fsdp_cfg: FSDPConfig = FSDPConfig(
            cpu_offload=False,
            tp_size=tp_size
        )
        engine = TrainEngine(
            model_cfg=dense_cfg, optim_cfg=optim_cfg, fsdp_cfg=fsdp_cfg
        )
        engine.from_hf(hf_path=QWEN3_PATH)

        loss_cfg = CELossConfig()

        total_steps = 100
        warmup_steps = total_steps * lr_cfg.warmup_ratio

        def warmup_fn(x):
            return x / warmup_steps if x < warmup_steps else 1

        lr_scheduler = LambdaLR(engine.optimizer, warmup_fn)

        tok = AutoTokenizer.from_pretrained(QWEN3_PATH)
        txt = "根据国际地球自转和参考系服务机构的数据，今年夏天是自2020年以来第六次地球自转加速。7月9日将成为有史以来最短的一天，比平时短1.3到1.6毫秒。 "
        input_ids = tok.encode(txt, return_tensors="pt").view(1, -1)
        labels = input_ids.clone()
        input_ids = input_ids[:, :-1]
        labels = labels[:, 1:]
        pack_len = 8192 - input_ids.shape[1]
        input_ids = pad_to_max_length(input_ids, 0, max_length=8192)
        labels = pad_to_max_length(labels, -100, max_length=8192)

        for _ in range(10):
            seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
            labels = labels.to(DEVICE)
            seq_ctx.num_padding = pack_len
            seq_ctx_list = [seq_ctx]
            LossContext = loss_cfg.loss_ctx_cls
            loss_ctx = loss_cfg.build(shifted_labels=labels, sp_mesh=None)
            loss_ctx_list = [loss_ctx]
            loss_ctx_list = LossContext.build_batches(loss_ctx_list)
            loss_ctx = loss_ctx_list[0]
            seq_ctx = seq_ctx_list[0]
            engine_input = [ModelItem(seq_ctx=seq_ctx, loss_ctx=loss_ctx)]
            loss_log, _ = engine.train_step(engine_input)
            grad_norm = engine.clip_grad_norm()
            engine.step_optimizer(grad_norm)
            lr_scheduler.step()

            embedding_weight = engine.model.embed_tokens.weight.full_tensor().mean().item()
            lm_head_weight = engine.model.lm_head.weight.full_tensor().mean().item()
            self.assertEqual(embedding_weight, lm_head_weight)

        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except:
            pass


    @parametrize.parametrize(
        "device,tp_size",
        [
            ("cuda", 1),
        ],
    )
    def test_qwen3vl_tie_embedding(self, device, tp_size):
        pg = self.create_pg(device)
        dense_cfg = Qwen3VLDense4BConfig()
        optim_cfg: AdamWConfig = AdamWConfig()
        lr_cfg: LRConfig = LRConfig(lr_min=1e-3)
        fsdp_cfg: FSDPConfig = FSDPConfig(
            cpu_offload=False,
            tp_size=tp_size
        )
        engine = VisionComposeTrainEngine(
            model_cfg=dense_cfg, optim_cfg=optim_cfg, fsdp_cfg=fsdp_cfg
        )
        engine.from_hf(hf_path=QWEN3_VL_DENSE_PATH)

        loss_cfg = CELossConfig()

        total_steps = 100
        warmup_steps = total_steps * lr_cfg.warmup_ratio

        def warmup_fn(x):
            return x / warmup_steps if x < warmup_steps else 1

        lr_scheduler = LambdaLR(engine.optimizer, warmup_fn)

        tok = AutoTokenizer.from_pretrained(QWEN3_VL_DENSE_PATH)
        image_str = '<|vision_start|><|image_pad|><|vision_end|>'
        txt = image_str+"根据国际地球自转和参考系服务机构的数据，今年夏天是自2020年以来第六次地球自转加速。7月9日将成为有史以来最短的一天，比平时短1.3到1.6毫秒。 "
        input_ids = tok.encode(txt, return_tensors="pt").view(1, -1)
        labels = input_ids.clone()
        input_ids = input_ids[:, :-1]
        labels = labels[:, 1:]
        pack_len = 8192 - input_ids.shape[1]
        input_ids = pad_to_max_length(input_ids, 0, max_length=8192)
        labels = pad_to_max_length(labels, -100, max_length=8192)

        for _ in range(10):
            seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)

            pixel_values = torch.randn(4, 1536, device='cuda', dtype=torch.bfloat16)
            image_grid_thw = torch.tensor([[1, 2, 2]], device='cuda')
            seq_ctx.pixel_values = pixel_values
            seq_ctx.image_grid_thw = image_grid_thw

            labels = labels.to(DEVICE)
            seq_ctx.num_padding = pack_len
            seq_ctx_list = [seq_ctx]
            LossContext = loss_cfg.loss_ctx_cls
            loss_ctx = loss_cfg.build(shifted_labels=labels, sp_mesh=None)
            loss_ctx_list = [loss_ctx]
            loss_ctx_list = LossContext.build_batches(loss_ctx_list)
            loss_ctx = loss_ctx_list[0]
            seq_ctx = seq_ctx_list[0]
            engine_input = [ModelItem(seq_ctx=seq_ctx, loss_ctx=loss_ctx)]
            loss_log, _ = engine.train_step(engine_input)
            grad_norm = engine.clip_grad_norm()
            engine.step_optimizer(grad_norm)
            lr_scheduler.step()

            embedding_weight = engine.model.language_model.embed_tokens.weight.full_tensor().mean().item()
            lm_head_weight = engine.model.language_model.lm_head.weight.full_tensor().mean().item()
            self.assertEqual(embedding_weight, lm_head_weight)

        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except:
            pass

    @parametrize.parametrize(
        "device,tp_size",
        [
            ("cuda", 1),
        ],
    )
    def test_save_hf(self, device, tp_size):
        self.create_pg(device)
        with torch.device("meta"):
            cfg = Qwen3Dense4BConfig()
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
