import os

import parametrize
import torch
from torch.testing._internal.common_distributed import DistributedTestBase
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from xtuner.v1.model.interns1 import InternS1Config, InternS1VisionConfig, InternS1ProjectorConfig
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.config import FSDPConfig
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.loss import CELossContext
from xtuner.v1.datasets.interns1_fn.process import build_transform,  dynamic_preprocess, preprocess_interns1
from xtuner.v1.utils.test_utils import init_data_mesh
from xtuner.v1.utils.pad import pad_to_multiple_of
from PIL import Image

# Intern-S1 30B A3
INTERNS1_MOE_PATH = os.environ["INTERNS1_MOE_PATH"]


class TestInternS1(DistributedTestBase):
    @parametrize.parametrize(
        "device,dispatcher,ep_size,sp_size, compile,tol",
        [
            ("cuda", "deepep", 8, 1, False, 1e-2),
            ("cuda", "all2all", 8, 1, False, 1e-2),
            ("cuda", None, 1, 1, False, 1e-2),
            ("cuda", "deepep", 8, 1, True, 4e-2),  # TODO: This test is flaky, need to fix it
            ("cuda", "all2all", 8, 2, False, 1e-2),
            ("cuda", None, 1, 2, False, 1e-2),
            ("cuda", "deepep", 8, 2, True, 4e-2),  # TODO: This test is flaky, need to fix it
        ],
    )
    def test_interns1_text_run(self, device, dispatcher, ep_size, sp_size, compile, tol):
        self.create_pg(device)
        if not compile:
            maybe_compile.clear_compile_targets()

        hf_config = AutoConfig.from_pretrained(
            INTERNS1_MOE_PATH,
            trust_remote_code=True,
        )
        hf_config.text_config.attn_implementation = "flash_attention_2"
        hf_config.vision_config.attn_implementation = "flash_attention_2"
        hf_model = AutoModelForCausalLM.from_pretrained(
            INTERNS1_MOE_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda"
        ).eval()  # avoid open drop_path

        tokenizer = AutoTokenizer.from_pretrained(INTERNS1_MOE_PATH, trust_remote_code=True)
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
            vision_cfg = InternS1VisionConfig()
            projector_cfg = InternS1ProjectorConfig()
            llm_cfg = Qwen3MoE30BA3Config(vocab_size=152967)
            llm_cfg.dispatcher = dispatcher
            llm_cfg.ep_size = ep_size
            model_cfg = InternS1Config(vision_config=vision_cfg, text_config=llm_cfg, projector_config=projector_cfg)
            interns1_model = model_cfg.build().to(torch.bfloat16)

        shift_input_ids = input_ids[:, :-1]
        shift_labels = input_ids[:, 1:]

        data_mesh = None
        if sp_size > 1:
            data_mesh = init_data_mesh(device, sp_size=sp_size)

        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to(device),))
        data_batch = [{'seq_ctx': seq_ctx, 'labels': shift_labels}]
        loss_ctx = CELossContext()
        data_batch = loss_ctx.build_list_ctx(data_batch, device=device, data_mesh=data_mesh)[0]
        interns1_model.from_hf(INTERNS1_MOE_PATH)
        interns1_model.eval()  # avoid open drop_path

        with torch.no_grad():
            output = interns1_model(
                seq_ctx=data_batch['seq_ctx'],
                loss_ctx=data_batch['loss_ctx'],
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=tol, rtol=tol))

    @parametrize.parametrize(
        "device,dispatcher,ep_size,sp_size,compile,tol",
        [
            ("cuda", "deepep", 8, 1, False, 1e-2),
            ("cuda", "all2all", 8, 1, False, 1e-2),
            ("cuda", None, 1, 1, False, 1e-2),
            ("cuda", "deepep", 8, 1, True, 4e-2),  # TODO: This test is flaky, need to fix it
            ("cuda", "all2all", 8, 2, False, 1e-2),
            ("cuda", None, 1, 2, False, 1e-2),
            ("cuda", "deepep", 8, 2, True, 4e-2),  # TODO: This test is flaky, need to fix it
        ],
    )
    def test_interns1_image_run(self, device, dispatcher, ep_size, sp_size, compile, tol):
        self.create_pg(device)
        if not compile:
            maybe_compile.clear_compile_targets()

        hf_config = AutoConfig.from_pretrained(
            INTERNS1_MOE_PATH,
            trust_remote_code=True,
        )
        hf_config.text_config.attn_implementation = "flash_attention_2"
        hf_config.vision_config.attn_implementation = "flash_attention_2"
        hf_model = AutoModelForCausalLM.from_pretrained(
            INTERNS1_MOE_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device
        ).eval()  # avoid open drop_path

        tokenizer = AutoTokenizer.from_pretrained(INTERNS1_MOE_PATH, trust_remote_code=True)

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
            vision_cfg = InternS1VisionConfig()
            projector_cfg = InternS1ProjectorConfig()
            llm_cfg = Qwen3MoE30BA3Config(vocab_size=152967)
            llm_cfg.dispatcher = dispatcher
            llm_cfg.ep_size = ep_size
            model_cfg = InternS1Config(vision_config=vision_cfg, text_config=llm_cfg, projector_config=projector_cfg)
            interns1_model = model_cfg.build().to(torch.bfloat16)

        shift_input_ids = input_ids[:, :-1]
        shift_labels = input_ids[:, 1:]

        data_mesh = None
        if sp_size > 1:
            data_mesh = init_data_mesh(device, sp_size=sp_size)

        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
        seq_ctx.image_flags = image_flags
        seq_ctx.pixel_values = pixel_values
        seq_ctx.to('cuda')
        data_batch = [{'seq_ctx': seq_ctx, 'labels': shift_labels}]
        loss_ctx = CELossContext()
        data_batch = loss_ctx.build_list_ctx(data_batch, device='cuda', data_mesh=data_mesh)[0]
        interns1_model.from_hf(INTERNS1_MOE_PATH)
        interns1_model.eval()  # avoid open drop_path

        with torch.no_grad():
            output = interns1_model(
                seq_ctx=data_batch['seq_ctx'],
                loss_ctx=data_batch['loss_ctx'],
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=tol, rtol=tol))

    @parametrize.parametrize(
        "device,dispatcher,ep_size,sp_size",
        [
            ("cuda", "all2all", 4, 1),
            ("cuda", "all2all", 8, 1),
            ("cuda", None, 1, 1),
            ("cuda", "all2all", 4, 2),
            ("cuda", "all2all", 8, 2),
            ("cuda", None, 1, 2),
        ],
    )
    def test_fsdp_text_accuracy(self, device, dispatcher, ep_size, sp_size):
        self.create_pg(device)
        maybe_compile.clear_compile_targets()
        hf_config = AutoConfig.from_pretrained(
            INTERNS1_MOE_PATH,
            trust_remote_code=True,
        )
        hf_config.text_config.attn_implementation = "flash_attention_2"
        hf_config.vision_config.attn_implementation = "flash_attention_2"
        hf_model = AutoModelForCausalLM.from_pretrained(
            INTERNS1_MOE_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda"
        ).eval()  # avoid open drop_path

        tokenizer = AutoTokenizer.from_pretrained(INTERNS1_MOE_PATH, trust_remote_code=True)
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
            vision_cfg = InternS1VisionConfig()
            projector_cfg = InternS1ProjectorConfig()
            llm_cfg = Qwen3MoE30BA3Config(vocab_size=152967)
            llm_cfg.dispatcher = dispatcher
            llm_cfg.ep_size = ep_size
            model_cfg = InternS1Config(vision_config=vision_cfg, text_config=llm_cfg, projector_config=projector_cfg)
            interns1_model = model_cfg.build().to(torch.bfloat16)

        fsdp_config = FSDPConfig(
            ep_size=ep_size,
            cpu_offload=False,
        )
        data_mesh = None
        if sp_size > 1:
            data_mesh = init_data_mesh(device, sp_size=sp_size)
        shift_input_ids = input_ids[:, :-1]
        shift_labels = input_ids[:, 1:]
        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
        data_batch = [{'seq_ctx': seq_ctx, 'labels': shift_labels}]
        loss_ctx = CELossContext()
        data_batch = loss_ctx.build_list_ctx(data_batch, device='cuda', data_mesh=data_mesh)[0]
        interns1_model.language_model.fully_shard(fsdp_config=fsdp_config)
        interns1_model.vision_tower.fully_shard(fsdp_config=fsdp_config)
        interns1_model.multi_modal_projector.fully_shard(fsdp_config=fsdp_config)
        interns1_model.fully_shard(fsdp_config=fsdp_config)

        interns1_model.from_hf(INTERNS1_MOE_PATH)
        interns1_model.eval()  # avoid open drop_path

        with torch.no_grad():
            output = interns1_model(
                seq_ctx=data_batch['seq_ctx'],
                loss_ctx=data_batch['loss_ctx'],
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=1e-2, rtol=1e-2))

    @parametrize.parametrize(
        "device,dispatcher,ep_size, sp_size",
        [
            ("cuda", "all2all", 4, 1),
            ("cuda", "all2all", 8, 1),
            ("cuda", None, 1, 1),
            ("cuda", "all2all", 4, 2),
            ("cuda", "all2all", 8, 2),
            ("cuda", None, 1, 2),
        ],
    )
    def test_fsdp_image_accuracy(self, device, dispatcher, ep_size, sp_size):
        self.create_pg(device)
        maybe_compile.clear_compile_targets()
        hf_config = AutoConfig.from_pretrained(
            INTERNS1_MOE_PATH,
            trust_remote_code=True,
        )
        hf_config.text_config.attn_implementation = "flash_attention_2"
        hf_config.vision_config.attn_implementation = "flash_attention_2"
        hf_model = AutoModelForCausalLM.from_pretrained(
            INTERNS1_MOE_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda"
        ).eval()  # avoid open drop_path

        tokenizer = AutoTokenizer.from_pretrained(INTERNS1_MOE_PATH, trust_remote_code=True)
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
            vision_cfg = InternS1VisionConfig()
            projector_cfg = InternS1ProjectorConfig()
            llm_cfg = Qwen3MoE30BA3Config(vocab_size=152967)
            llm_cfg.dispatcher = dispatcher
            llm_cfg.ep_size = ep_size
            model_cfg = InternS1Config(vision_config=vision_cfg, text_config=llm_cfg, projector_config=projector_cfg)
            interns1_model = model_cfg.build().to(torch.bfloat16)

        fsdp_config = FSDPConfig(
            ep_size=ep_size,
            cpu_offload=False,
        )
        data_mesh = None
        if sp_size > 1:
            data_mesh = init_data_mesh(device, sp_size=sp_size)

        shift_input_ids = input_ids[:, :-1]
        shift_labels = input_ids[:, 1:]
        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
        seq_ctx.image_flags = image_flags
        seq_ctx.pixel_values = pixel_values
        seq_ctx.to('cuda')
        data_batch = [{'seq_ctx': seq_ctx, 'labels': shift_labels}]
        loss_ctx = CELossContext()
        data_batch = loss_ctx.build_list_ctx(data_batch, device='cuda', data_mesh=data_mesh)[0]
        interns1_model.language_model.fully_shard(fsdp_config=fsdp_config)
        interns1_model.vision_tower.fully_shard(fsdp_config=fsdp_config)
        interns1_model.multi_modal_projector.fully_shard(fsdp_config=fsdp_config)
        interns1_model.fully_shard(fsdp_config=fsdp_config)

        interns1_model.from_hf(INTERNS1_MOE_PATH)
        interns1_model.eval()  # avoid open drop_path

        with torch.no_grad():
            output = interns1_model(
                seq_ctx=data_batch['seq_ctx'],
                loss_ctx=data_batch['loss_ctx'],
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=1e-2, rtol=1e-2))

    @parametrize.parametrize(
        "device,dispatcher,ep_size",
        [
            ("cuda", None, 1),
            ("cuda", "all2all", 4),
            ("cuda", "all2all", 8),
        ],
    )
    def test_save_hf(self, device, dispatcher, ep_size):
        pass

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
