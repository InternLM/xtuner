import os

import parametrize
import torch
from torch.testing._internal.common_distributed import DistributedTestBase
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch.distributed as dist
import tempfile
from pathlib import Path
import json
from safetensors import safe_open

from xtuner.v1.model.interns1 import InternS1MiniConfig
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContextInputItem
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.model.dense.qwen3 import Qwen3Dense8BConfig
from xtuner.v1.config import FSDPConfig
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.datasets.interns1_fn.process import build_transform,  dynamic_preprocess, preprocess_interns1
from xtuner.v1.utils.test_utils import init_data_mesh
from PIL import Image

# Intern-S1-mini
INTERNS1_DENSE_PATH = os.environ["INTERNS1_DENSE_PATH"]


class TestInternS1(DistributedTestBase):
    @parametrize.parametrize(
        "device,tol",
        [
            ("cuda", 1e-2),
        ],
    )
    def test_interns1_text_run(self, device, tol):
        self.create_pg(device)
        if not compile:
            maybe_compile.clear_compile_targets()

        hf_config = AutoConfig.from_pretrained(
            INTERNS1_DENSE_PATH,
            trust_remote_code=True,
        )
        hf_config.text_config.attn_implementation = "flash_attention_2"
        hf_config.vision_config.attn_implementation = "flash_attention_2"
        hf_model = AutoModelForCausalLM.from_pretrained(
            INTERNS1_DENSE_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda"
        ).eval()  # avoid open drop_path

        tokenizer = AutoTokenizer.from_pretrained(INTERNS1_DENSE_PATH, trust_remote_code=True)
        input_ids = tokenizer("吃葡萄不吐葡萄皮", return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            output = hf_model(
                input_ids=input_ids,
                labels=input_ids.clone(),
            )
        expected_loss = output.loss

        del hf_model
        torch.cuda.empty_cache()

        with torch.device("meta"):
            model_cfg = InternS1MiniConfig()
            interns1_model = model_cfg.build().to(torch.bfloat16)
        
        interns1_model.from_hf(INTERNS1_DENSE_PATH)
        interns1_model.eval()  # avoid open drop_path
        
        loss_cfg = CELossConfig()

        shift_input_ids = input_ids[:, :-1]
        shifted_labels = input_ids[:, 1:]

        data_mesh = None
        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to(device),))
        
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

        with torch.no_grad():
            output = interns1_model(
                seq_ctx=seq_ctx,
                loss_ctx=loss_ctx,
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=tol, rtol=tol))

    @parametrize.parametrize(
        "device,sp_size,tol",
        [
            ("cuda", 1, 1e-2),
            ("cuda", 2, 1e-2),
        ],
    )
    def test_interns1_image_run(self, device, sp_size, tol):
        self.create_pg(device)
        maybe_compile.clear_compile_targets()

        hf_config = AutoConfig.from_pretrained(
            INTERNS1_DENSE_PATH,
            trust_remote_code=True,
        )
        hf_config.text_config.attn_implementation = "flash_attention_2"
        hf_config.vision_config.attn_implementation = "flash_attention_2"
        hf_model = AutoModelForCausalLM.from_pretrained(
            INTERNS1_DENSE_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device
        ).eval()  # avoid open drop_path

        tokenizer = AutoTokenizer.from_pretrained(INTERNS1_DENSE_PATH, trust_remote_code=True)

        conversations = [{"from": "human", "value": '<image>\nPlease describe the image shortly.'}]
        image_path = 'tests/resource/mscoco_twocat_000000039769.jpg'
        image = Image.open(image_path).convert("RGB")
        images = dynamic_preprocess(
            image,
            min_num=1,
            max_num=12,
            image_size=448,
            use_thumbnail=True,
        )
        transform = build_transform(
            is_train=False,
            input_size=448,
            pad2square=False,
            normalize_type="imagenet"
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)

        ret = preprocess_interns1(
            [conversations],
            tokenizer,
            [256 * num_patches]
        )
        input_ids = torch.tensor(ret["input_ids"])[None].cuda()
        image_flags = torch.tensor([1] * num_patches, dtype=torch.long).cuda()
        pixel_values = pixel_values.to(device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            output = hf_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                labels=input_ids.clone(),
            )
        expected_loss = output.loss

        del hf_model
        torch.cuda.empty_cache()

        with torch.device("meta"):
            model_cfg = InternS1MiniConfig()
            interns1_model = model_cfg.build().to(torch.bfloat16)
        
        interns1_model.from_hf(INTERNS1_DENSE_PATH)
        interns1_model.eval()  # avoid open drop_path

        loss_cfg = CELossConfig()
        
        shift_input_ids = input_ids[:, :-1]
        shifted_labels = input_ids[:, 1:]

        data_mesh = None
        sp_mesh = None
        if sp_size > 1:
            data_mesh = init_data_mesh(device, sp_size=sp_size)
            sp_mesh = data_mesh["sp"]

        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
        seq_ctx.image_flags = image_flags
        seq_ctx.pixel_values = pixel_values
        seq_ctx.to('cuda')
        loss_ctx_input = CELossContextInputItem(shifted_labels=shifted_labels)
        loss_ctx_input = loss_ctx_input.to('cuda')

        if sp_size > 1:
            seq_ctx = seq_ctx.split(sp_mesh)
            loss_ctx_input = loss_ctx_input.sp_split(sp_mesh)

        seq_ctx_list = [seq_ctx]
        loss_ctx_input_list: list[CELossContextInputItem] = [loss_ctx_input]

        LossContext = loss_cfg.loss_ctx_cls
        batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
            loss_ctx_input_list, 
            loss_cfg,
        )
        loss_kwargs = batches_loss_kwargs[0]
        loss_ctx = LossContext(loss_cfg, loss_kwargs)
        seq_ctx = seq_ctx_list[0]

        with torch.no_grad():
            output = interns1_model(
                seq_ctx=seq_ctx,
                loss_ctx=loss_ctx,
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=tol, rtol=tol))

    @parametrize.parametrize(
        "device,tol",
        [
            ("cuda", 1e-2),
        ],
    )
    def test_fsdp_text_accuracy(self, device, tol):
        self.create_pg(device)
        maybe_compile.clear_compile_targets()
        hf_config = AutoConfig.from_pretrained(
            INTERNS1_DENSE_PATH,
            trust_remote_code=True,
        )
        hf_config.text_config.attn_implementation = "flash_attention_2"
        hf_config.vision_config.attn_implementation = "flash_attention_2"
        hf_model = AutoModelForCausalLM.from_pretrained(
            INTERNS1_DENSE_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda"
        ).eval()  # avoid open drop_path

        tokenizer = AutoTokenizer.from_pretrained(INTERNS1_DENSE_PATH, trust_remote_code=True)
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
            model_cfg = InternS1MiniConfig()
            interns1_model = model_cfg.build().to(torch.bfloat16)
        
        fsdp_config = FSDPConfig(
            cpu_offload=False,
        )
        data_mesh = None

        interns1_model.language_model.fully_shard(fsdp_config=fsdp_config)
        interns1_model.vision_tower.fully_shard(fsdp_config=fsdp_config)
        interns1_model.multi_modal_projector.fully_shard(fsdp_config=fsdp_config)
        interns1_model.fully_shard(fsdp_config=fsdp_config)
        
        interns1_model.from_hf(INTERNS1_DENSE_PATH)
        interns1_model.eval()  # avoid open drop_path

        shift_input_ids = input_ids[:, :-1]
        shifted_labels = input_ids[:, 1:]
        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
        loss_ctx_input = CELossContextInputItem(shifted_labels=shifted_labels)
        loss_ctx_input = loss_ctx_input.to('cuda')

        seq_ctx_list = [seq_ctx]
        loss_ctx_input_list: list[CELossContextInputItem] = [loss_ctx_input]

        loss_cfg = CELossConfig()
        LossContext = loss_cfg.loss_ctx_cls
        batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
            loss_ctx_input_list, 
            loss_cfg,
        )
        loss_kwargs = batches_loss_kwargs[0]
        loss_ctx = LossContext(loss_cfg, loss_kwargs)
        seq_ctx = seq_ctx_list[0]

        with torch.no_grad():
            output = interns1_model(
                seq_ctx=seq_ctx,
                loss_ctx=loss_ctx,
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=tol, rtol=tol))

    @parametrize.parametrize(
        "device,sp_size, compile, tol",
        [
            ("cuda", 1, False, 1e-2),
            ("cuda", 2, False, 1e-2),
            ("cuda", 1, True, 1e-2),
            ("cuda", 2, True, 1e-2),
        ],
    )
    def test_fsdp_image_accuracy(self, device, sp_size, compile, tol):
        self.create_pg(device)
        if not compile:
            maybe_compile.clear_compile_targets()

        hf_config = AutoConfig.from_pretrained(
            INTERNS1_DENSE_PATH,
            trust_remote_code=True,
        )
        hf_config.text_config.attn_implementation = "flash_attention_2"
        hf_config.vision_config.attn_implementation = "flash_attention_2"
        hf_model = AutoModelForCausalLM.from_pretrained(
            INTERNS1_DENSE_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda"
        ).eval()  # avoid open drop_path

        tokenizer = AutoTokenizer.from_pretrained(INTERNS1_DENSE_PATH, trust_remote_code=True)
        conversations = [{"from": "human", "value": '<image>\nPlease describe the image shortly.'}]
        image_path = 'tests/resource/mscoco_twocat_000000039769.jpg'
        image = Image.open(image_path).convert("RGB")
        images = dynamic_preprocess(
            image,
            min_num=1,
            max_num=12,
            image_size=448,
            use_thumbnail=True,
        )
        transform = build_transform(
            is_train=False,
            input_size=448,
            pad2square=False,
            normalize_type="imagenet"
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)

        ret = preprocess_interns1(
            [conversations],
            tokenizer,
            [256 * num_patches]
        )
        input_ids = torch.tensor(ret["input_ids"])[None].cuda()
        image_flags = torch.tensor([1] * num_patches, dtype=torch.long).cuda()
        pixel_values = pixel_values.to(device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            output = hf_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                labels=input_ids.clone(),
            )
        expected_loss = output.loss

        del hf_model
        torch.cuda.empty_cache()

        with torch.device("meta"):
            model_cfg = InternS1MiniConfig()
            interns1_model = model_cfg.build().to(torch.bfloat16)

        fsdp_config = FSDPConfig(
            cpu_offload=False,
        )
        data_mesh = None
        sp_mesh = None
        if sp_size > 1:
            data_mesh = init_data_mesh(device, sp_size=sp_size)
            sp_mesh = data_mesh["sp"]
        
        interns1_model.language_model.fully_shard(fsdp_config=fsdp_config)
        interns1_model.vision_tower.fully_shard(fsdp_config=fsdp_config)
        interns1_model.multi_modal_projector.fully_shard(fsdp_config=fsdp_config)
        interns1_model.fully_shard(fsdp_config=fsdp_config)

        interns1_model.from_hf(INTERNS1_DENSE_PATH)
        interns1_model.eval()  # avoid open drop_path

        shift_input_ids = input_ids[:, :-1]
        shifted_labels = input_ids[:, 1:]
        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
        seq_ctx.image_flags = image_flags
        seq_ctx.pixel_values = pixel_values
        seq_ctx.to('cuda')
        loss_ctx_input = CELossContextInputItem(shifted_labels=shifted_labels)
        loss_ctx_input = loss_ctx_input.to('cuda')

        if sp_size > 1:
            seq_ctx = seq_ctx.split(sp_mesh)
            loss_ctx_input = loss_ctx_input.sp_split(sp_mesh)

        seq_ctx_list = [seq_ctx]
        loss_ctx_input_list: list[CELossContextInputItem] = [loss_ctx_input]

        loss_cfg = CELossConfig()
        LossContext = loss_cfg.loss_ctx_cls
        batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
            loss_ctx_input_list, 
            loss_cfg,
        )
        loss_kwargs = batches_loss_kwargs[0]
        loss_ctx = LossContext(loss_cfg, loss_kwargs)
        seq_ctx = seq_ctx_list[0]

        with torch.no_grad():
            output = interns1_model(
                seq_ctx=seq_ctx,
                loss_ctx=loss_ctx,
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=tol, rtol=tol))

    @parametrize.parametrize(
        "device,tp_size",
        [
            ("cuda", 1),
        ],
    )
    def test_save_hf(self, device, tp_size):
        self.create_pg(device)
        with torch.device("meta"):
            model_cfg = InternS1MiniConfig()
            interns1_model = model_cfg.build().to(torch.bfloat16)

        fsdp_config = FSDPConfig(
            tp_size=tp_size,
            cpu_offload=False,
        )

        cache_save_fh = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            syncdir = [tmpdir]
            dist.broadcast_object_list(syncdir, src=0)
            tmpdir = Path(syncdir[0])
            interns1_model.language_model.fully_shard(fsdp_config=fsdp_config)
            interns1_model.vision_tower.fully_shard(fsdp_config=fsdp_config)
            interns1_model.multi_modal_projector.fully_shard(fsdp_config=fsdp_config)
            interns1_model.fully_shard(fsdp_config=fsdp_config)
            interns1_model.from_hf(INTERNS1_DENSE_PATH)
            interns1_model.save_hf(tmpdir)

            origin_hf_path = Path(INTERNS1_DENSE_PATH)
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
